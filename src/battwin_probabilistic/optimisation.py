"""Capacity optimisation with unchanged support, cap and physical gates."""
from dataclasses import asdict
from pathlib import Path
import hashlib,json,time,inspect
import numpy as np
import pandas as pd
from . import design_space as old
from .design_space import *
from .scrap import ScrapPolicyMixin
from .config import emit,project_root,SimulationConfig
from .artifacts import sha256
from .chain import ProcessChain
from .models import ModelCatalog
from .physics import physical_stage_flags
from .inverse_design import InverseDesignMixin
from .formulation import COMPONENTS,SOLIDS,IDENTITIES,key as formulation_key,prepare_formulation,with_calculated_solids,validate_identities

def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')

class SearchExperiment(ScrapPolicyMixin):
    def __init__(self,development,config=None,catalog=None,output_dir=None,method='kuleshov',callback=None):
        self.config=config or SearchConfig()
        self.method=method;self.callback=callback
        if method not in ['kuleshov','uncalibrated']:raise ValueError('Use kuleshov or uncalibrated')
        self.catalog=catalog or ModelCatalog()
        self.out=Path(output_dir or project_root()/'runs/design'/f'{self.config.discharge_rate:g}C_{method}_{"capped" if self.config.apply_capacity_caps else "uncapped"}').resolve()
        for name in ['reports','figures','cache']:(self.out/name).mkdir(parents=True,exist_ok=True)
        self.line=ProcessChain(self.catalog,SimulationConfig(calibration=method))
        from .orchestration import core
        self.chain=core.__dict__
        self.context=dict(physical_stage_flags=physical_stage_flags,align_chain_input=self.line.align_inputs)
        self.bundles={key:self.catalog.load(key,method).artifact_path for key in ['mixing','coating_drying','calendaring','electrochemical']}
        if 'partition' in development and development.partition.astype(str).str.lower().eq('test').any():
            raise ValueError('Test data cannot define the design space or capacity caps')
        development=self.line.align_inputs(development)
        self.base=development.loc[development.Chemistry.eq(self.config.chemistry)&np.isclose(
            development['Discharge rate (*C)'],self.config.discharge_rate,rtol=0,atol=1e-10)].copy()
        if self.base.empty:raise ValueError('No development support for chemistry/discharge rate')
        self.ipi_support={c:(float(self.base[c].min()),float(self.base[c].max())) for c in
            ['Mass loading dry (mg/cm2)','Calendered thickness (um)','Calendered porosity (%)']}
        self.capacity_bounds={c:float(self.base[c].max()) for c in [GRAV,VOL]}
        self.capacity_reference=dict(chemistry=self.config.chemistry,discharge_rate=self.config.discharge_rate,
            development_samples=len(self.base),source='explicit development frame; no Test data',maxima=self.capacity_bounds)
        write(self.out/'capacity_reference.json',self.capacity_reference)
        self.required=self.line.get_required_input_columns()
        generated={c for outputs in OUTPUTS.values() for c in outputs}|{'Viscosity (Pa.s)'}
        if set(self.required)&generated:raise ValueError('Generated model outputs cannot be design controls')
        self.base=self.base[self.required].drop_duplicates().reset_index(drop=True)
        self.bounds={c:(float(self.base[c].min()),float(self.base[c].max())) for c in CONTINUOUS}
        component_bounds=self.config.formulation_bounds
        if any(formulation_key(c)==formulation_key(SOLIDS) for c in component_bounds):
            raise ValueError('Solid content is calculated. Specify component formulation_bounds instead.')
        if set(component_bounds)-set(COMPONENTS):raise ValueError('Unknown formulation component in formulation_bounds.')
        if component_bounds and not self.config.vary_formulation:raise ValueError('Enable vary_formulation to use formulation_bounds.')
        if self.config.vary_formulation:
            for c in COMPONENTS:
                observed=(float(self.base[c].min()),float(self.base[c].max()))
                bounds=component_bounds.get(c,observed)
                if len(bounds)!=2 or not np.isfinite(bounds).all() or not 0<=bounds[0]<=bounds[1]<=100:
                    raise ValueError('Invalid percentage bounds for '+c)
                lo,hi=map(float,bounds)
                if not self.config.allow_formulation_extrapolation and (lo<observed[0]-1e-10 or hi>observed[1]+1e-10):
                    raise ValueError('Formulation bounds outside observed support for '+c+'; explicitly enable allow_formulation_extrapolation to proceed.')
                self.bounds[c]=(lo,hi)
                if lo>0 and c in IDENTITIES and self.base[IDENTITIES[c]].map(lambda v:str(v).strip(' \"\'').casefold() in ['none','not used','nan','']).all():
                    raise ValueError('No material identity is available for positive '+c+' in the supplied development profiles.')
            minimum=sum(self.bounds[c][0] for c in COMPONENTS);maximum=sum(self.bounds[c][1] for c in COMPONENTS)
            if minimum>100 or maximum<=0:raise ValueError('The formulation bounds contain no positive composition totalling at most 100%.')
        self.solid_support=(float(self.base[SOLIDS].min()),float(self.base[SOLIDS].max()))
        self.levels={c:sorted(self.base[c].unique().tolist()) for c in DISCRETE}
        self.rng=np.random.default_rng(self.config.seed);self.next_id=0;self.seen=set()
        self.all_paths=pd.DataFrame();self.refined=pd.DataFrame();self.alternatives=pd.DataFrame()
        self.verification_files=[];self.target=None;self.shared_cache=self.out/'cache'
        code_files=['chain.py','distributions.py','calibration.py','numerics.py','orchestration/core.py',
            'optimisation.py','inverse_design.py','design_space.py','formulation.py']
        package=Path(__file__).resolve().parent
        signature=dict(config=asdict(self.config),method=method,
            development_sha256=hashlib.sha256(pd.util.hash_pandas_object(development,index=True).values.tobytes()).hexdigest(),
            bundles={key:sha256(path) for key,path in self.bundles.items()},
            implementation={name:sha256(package/name) for name in code_files})
        self.signature=hashlib.sha256(json.dumps(signature,sort_keys=True,default=str).encode()).hexdigest()
        write(self.out/'config.json',signature)
        domain=[]
        for c in self.required:
            numeric=pd.api.types.is_numeric_dtype(self.base[c])
            mode='continuous' if c in self.bounds else 'discrete' if c in DISCRETE else 'derived' if c in ['Shear rate',SOLIDS] else 'observed recipe/profile block'
            lo,hi=self.bounds.get(c,(self.base[c].min(),self.base[c].max()))
            domain.append(dict(parameter=c,mode=mode,minimum=lo if numeric else None,
                maximum=hi if numeric else None,observed_values=json.dumps(self.base[c].unique().tolist(),default=str)))
        self.save(pd.DataFrame(domain),'design_space')

    def prepare_candidate(self,row):
        """Calculate solids without rebalancing; reject inadmissible proposals."""
        try:
            values=with_calculated_solids(row);validate_identities(values)
        except ValueError:return None
        if not self.config.allow_formulation_extrapolation and not self.solid_support[0]-1e-9<=values[SOLIDS]<=self.solid_support[1]+1e-9:
            return None
        return pd.Series(values)

    def _log(self,*args,**kwargs):
        emit(self.callback,'design_progress',message=' '.join(str(v) for v in args))

    def save(self, table, name):
        table.to_csv(self.out / 'reports' / (name + '.csv'), index=False)
        return table

    def generate(self, n, parents=None, local=False):
        """Continuous operating settings, with valid observed recipe/profile blocks."""
        rows = []
        parents = self.base if parents is None or parents.empty else parents
        for _ in range(n * 100 + 1):
            row = parents.iloc[int(self.rng.integers(len(parents)))][self.required].copy()
            anchor = row.copy()
            for c, (lo, hi) in self.bounds.items():
                if local:
                    row[c] = np.clip(float(row[c]) + self.rng.normal(0, 0.1 * (hi - lo)), lo, hi)
                else:
                    row[c] = self.rng.uniform(lo, hi)
            for c, levels in self.levels.items():
                if not local or self.rng.random() < 0.2:
                    row[c] = self.rng.choice(levels)
            row['Shear rate'] = derived_shear(anchor, row['Coating Speed (m/min)'], row['Coating Gap (micron)'], row['Coating ratio (%)'])
            shear_lo, shear_hi = (float(self.base['Shear rate'].min()), float(self.base['Shear rate'].max()))
            if not shear_lo - 1e-06 <= row['Shear rate'] <= shear_hi + 1e-06:
                continue
            if float(row['Calendaring Gap (um)']) >= float(row['Coating Gap (micron)']):
                continue
            row=self.prepare_candidate(row)
            if row is None:continue
            key = tuple((str(row[c]) for c in self.required))
            if key in self.seen:
                continue
            self.seen.add(key)
            row['path_id'] = self.next_id
            self.next_id += 1
            rows.append(row)
            if len(rows) == n:
                break
        if not rows:
            raise ValueError('No new feasible design specifications generated')
        return pd.DataFrame(rows).reset_index(drop=True)

    def observed_paths(self):
        rows = self.base.copy()
        rows['Shear rate'] = [derived_shear(r, r['Coating Speed (m/min)'], r['Coating Gap (micron)'], r['Coating ratio (%)']) for _, r in rows.iterrows()]
        rows = rows.drop_duplicates().reset_index(drop=True)
        rows = rows[rows['Calendaring Gap (um)'] < rows['Coating Gap (micron)']].copy()
        rows['path_id'] = np.arange(self.next_id, self.next_id + len(rows))
        self.next_id += len(rows)
        self.seen.update((tuple((str(row[c]) for c in self.required)) for _, row in rows.iterrows()))
        return rows.reset_index(drop=True)

    def simulate(self, specs, n, seed):
        specs=prepare_formulation(specs)
        _, particles = self.line.simulate_items_particles(specs[self.required].reset_index(drop=True), n_particles_override=n, n_member_repeats=max(1, n // 20), random_state=int(seed), final_kpi_columns=[GRAV, VOL], spec_limits={}, quantile_grid=self.chain['SAMPLE_QUANTILE_GRID'], n_quantile_samples=1000, keep_particles=True)
        assert len(particles) == len(specs) * n
        particles['path_id'] = specs.path_id.to_numpy()[particles.scenario_index.to_numpy(int)]
        return particles

    def summarise(self, specs, particles, phase, tail_limit):
        specs=prepare_formulation(specs)
        physical, _ = self.context['physical_stage_flags'](particles, self.line.stages)
        all_physical = ~np.logical_or.reduce(list(physical.values()))
        work = particles.copy()
        work['physical_pass'] = all_physical
        group = work.groupby('path_id', sort=False)
        result = specs.set_index('path_id').copy()
        for col in [c for v in OUTPUTS.values() for c in v]:
            result[col + '__mean'] = group[col].mean()
            result[col + '__q05'] = group[col].quantile(0.05)
            result[col + '__q95'] = group[col].quantile(0.95)
        result['physical_pass'] = group.physical_pass.mean()
        for col, label in [(GRAV, 'grav'), (VOL, 'vol')]:
            result[label + '_max_particle_share'] = group[col].max() / group[col].sum()
            result[label + '_mean_mc_se'] = group[col].std() / np.sqrt(group.size())
        result['n_particles'] = group.size()
        result['phase'] = phase
        result = self.apply_eligibility(result, tail_limit)
        return result.reset_index()

    def apply_eligibility(self, result, tail_limit, enforce_capacity_caps=None):
        support = np.ones(len(result), bool)
        for c, (lo, hi) in self.ipi_support.items():
            support &= result[c + '__mean'].between(lo, hi).to_numpy()
        result['ipi_support_ok'] = support
        bounded = np.ones(len(result), bool)
        for c, upper in self.capacity_bounds.items():
            bounded &= (result[c + '__mean'] <= upper + 1e-10).to_numpy()
        result['capacity_bounds_ok'] = bounded
        enforce = self.config.apply_capacity_caps if enforce_capacity_caps is None else enforce_capacity_caps
        result['eligible'] = support & (bounded if enforce else True) & (result.physical_pass >= self.config.min_physical_pass) & (result.grav_max_particle_share <= tail_limit) & (result.vol_max_particle_share <= tail_limit) & np.isfinite(result[[GRAV + '__mean', VOL + '__mean']]).all(axis=1) & (result[[GRAV + '__mean', VOL + '__mean']] > 0).all(axis=1)
        return result

    def evaluate(self, specs, n, seed, phase, tail_limit):
        specs=prepare_formulation(specs)
        summaries = []
        for start in range(0, len(specs), self.config.batch_size):
            batch = specs.iloc[start:start + self.config.batch_size].copy()
            key = hashlib.sha256((self.signature + phase + str(n) + str(seed)).encode() + pd.util.hash_pandas_object(batch[self.required + ['path_id']], index=False).to_numpy().tobytes()).hexdigest()
            cache = self.out / 'cache' / (key + '.pkl.gz')
            t = time.monotonic()
            if cache.exists():
                summary = self.apply_eligibility(pd.read_pickle(cache), tail_limit)
            else:
                particles = self.simulate(batch, n, seed)
                summary = self.summarise(batch, particles, phase, tail_limit)
                summary.to_pickle(cache, compression='gzip')
            summaries.append(summary)
            self._log(f'{phase}: {min(start + len(batch), len(specs))}/{len(specs)} paths, {n} particles/path, {time.monotonic() - t:.1f}s', flush=True)
        return pd.concat(summaries, ignore_index=True)

    def front(self, frame):
        valid = frame.loc[frame.eligible].copy()
        if valid.empty:
            raise ValueError('No tail-stable, physically supported candidate; inspect searched_paths.csv')
        return valid.loc[pareto_mask(valid[[GRAV + '__mean', VOL + '__mean']])].copy()

    def optimise(self, shared_paths=None):
        cfg = self.config
        if shared_paths is None:
            initial = pd.concat([self.observed_paths(), self.generate(cfg.initial_random)], ignore_index=True)
            self.all_paths = self.evaluate(initial, cfg.screen_particles, cfg.seed, 'initial', cfg.max_tail_share_screen)
            self.save(self.all_paths, 'searched_paths')
            for gen in range(cfg.generations):
                parents = []
                for enforce in [False, True]:
                    view = self.apply_eligibility(self.all_paths.copy(), cfg.max_tail_share_screen, enforce)
                    if view.eligible.any():
                        parents.append(self.front(view))
                parents = pd.concat(parents, ignore_index=True).drop_duplicates('path_id')
                candidates = self.generate(cfg.offspring, parents, local=True)
                evaluated = self.evaluate(candidates, cfg.screen_particles, cfg.seed + gen + 1, f'pareto_generation_{gen + 1}', cfg.max_tail_share_screen)
                self.all_paths = pd.concat([self.all_paths, evaluated], ignore_index=True)
                self.save(self.all_paths, 'searched_paths')
        else:
            self.all_paths = self.apply_eligibility(shared_paths.copy(), cfg.max_tail_share_screen)
            self.next_id = int(self.all_paths.path_id.max()) + 1
            self.seen.update((tuple((str(row[c]) for c in self.required)) for _, row in self.all_paths.iterrows()))
            self.save(self.all_paths, 'searched_paths')
        front = self.front(self.all_paths)
        eligible = self.all_paths.loc[self.all_paths.eligible].copy()
        objectives = eligible[[GRAV + '__mean', VOL + '__mean']].to_numpy()
        span = np.maximum(np.ptp(objectives, axis=0), 1e-09)
        eligible['utopia_distance'] = np.linalg.norm((objectives.max(axis=0) - objectives) / span, axis=1)
        priority = pd.concat([front, eligible.sort_values('utopia_distance')]).drop_duplicates('path_id').head(cfg.max_refine)
        self.refined = self.evaluate(priority, cfg.refine_particles, cfg.seed + 100, 'front_refinement', cfg.max_tail_share_refine)
        self.save(self.refined, 'refined_paths')
        front = self.refined.loc[self.refined.eligible].copy()
        if front.empty:
            raise ValueError('No refined candidate satisfies the objective/support bounds')
        confirmed_blocks = []
        for start in range(0, len(front), self.config.batch_size):
            specs = front.iloc[start:start + self.config.batch_size].copy()
            runs = []
            for offset in [1001, 2001]:
                t = time.monotonic()
                runs.append(self.simulate(specs, cfg.verify_particles_per_seed, cfg.seed + offset))
                self._log(f'Pareto confirmation: {start + len(specs)}/{len(front)} points, seed {offset}, {time.monotonic() - t:.1f}s', flush=True)
            confirmed_blocks.append(self.summarise(specs, pd.concat(runs, ignore_index=True), 'front_confirmation', cfg.max_tail_share_verify))
        confirmed = pd.concat(confirmed_blocks, ignore_index=True)
        self.save(pd.concat([self.refined, confirmed], ignore_index=True), 'refined_paths')
        self.refined = confirmed
        final_front = self.front(confirmed)
        knee, method = balanced_knee(final_front[[GRAV + '__mean', VOL + '__mean']])
        chosen = final_front.iloc[knee]
        spec = chosen.to_frame().T
        self.target = chosen[[GRAV + '__mean', VOL + '__mean']].to_numpy(float)
        self.save(final_front, 'pareto_front')
        target_info = dict(path_id=int(chosen.path_id), selection=method, gravimetric=float(self.target[0]), volumetric=float(self.target[1]), tolerance=cfg.target_tolerance, objective_upper_bounds=self.capacity_bounds, capacity_caps_applied=cfg.apply_capacity_caps, lower=(self.target * (1 - cfg.target_tolerance)).tolist(), upper=(self.target * (1 + cfg.target_tolerance)).tolist(), caveat='Approximate sampled Pareto front, not a proven global optimum')
        (self.out / 'selected_target.json').write_text(json.dumps(target_info, indent=2), encoding='utf-8')
        self.target_spec = spec
        self._log('Confirmed capacity target:', self.target.tolist(), method, flush=True)
        return (final_front, target_info)

class Experiment(InverseDesignMixin,SearchExperiment):
    """Current Kuleshov/raw search policies, supplied with explicit resources."""
    def simulate(self,specs,n,seed):
        result=super().simulate(specs,n,seed)
        result['verification_seed']=seed
        return result

    def evaluate(self,specs,n,seed,phase,tail_limit):
        specs=prepare_formulation(specs)
        summaries=[]
        batch_size=8 if phase=='initial' or phase.startswith('pareto_generation') else self.config.batch_size
        for start in range(0,len(specs),batch_size):
            batch=specs.iloc[start:start+batch_size].copy()
            payload=pd.util.hash_pandas_object(batch[self.required+['path_id']],index=False).to_numpy().tobytes()
            key=hashlib.sha256((self.signature+str(n)+str(seed)).encode()+payload).hexdigest()
            cache=self.shared_cache/(key+'.pkl.gz')
            if cache.exists():summary=pd.read_pickle(cache)
            else:
                summary=self.summarise(batch,self.simulate(batch,n,seed),phase,tail_limit)
                summary.to_pickle(cache,compression='gzip')
            summary['phase']=phase;summary=self.apply_eligibility(summary,tail_limit)
            summaries.append(summary)
            emit(self.callback,'design_evaluation',phase=phase,completed=min(start+len(batch),len(specs)),total=len(specs),particles=n)
        if not summaries:raise ValueError('No candidate specifications to evaluate')
        return pd.concat(summaries,ignore_index=True)

def optimise_capacity(development,config=None,catalog=None,output_dir=None,method='kuleshov',callback=None):
    experiment=Experiment(development,config,catalog,output_dir,method,callback)
    try:
        experiment.optimise()
        status='confirmed_target'
    except ValueError as exc:
        feasibility_errors=('No tail-stable, physically supported candidate','No refined candidate satisfies','No objects to concatenate')
        if experiment.all_paths.empty or not str(exc).startswith(feasibility_errors):raise
        status='confirmed_target' if experiment.broader_feasibility_search(experiment.all_paths) else 'no_confirmed_feasible_target'
    write(experiment.out/'run_status.json',dict(status=status,target=experiment.target,
        capped=experiment.config.apply_capacity_caps,method=method,criteria_relaxed=False))
    return experiment
