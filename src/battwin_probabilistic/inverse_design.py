"""Existing multiscale inverse search and two-seed acceptance checks."""
import hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
from . import design_space as old
from .design_space import GRAV,VOL,CONTINUOUS,DISCRETE,CONTROL_NAMES,derived_shear
from .formulation import prepare_formulation

def write(path,value):
    Path(path).write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8')

def diverse_nearest(experiment, frame, n):
    """Target proximity plus control-space diversity; preserve nearest points."""
    frame = frame.copy()
    frame["target_distance"] = np.max(np.abs(frame[[GRAV+"__mean", VOL+"__mean"]].to_numpy()/experiment.target-1),axis=1)
    frame = frame.sort_values(["target_distance", "path_id"])
    if len(frame) <= n:
        return frame
    # Half by distance, half spanning a near-target neighbourhood. Diversity
    # affects which candidates are simulated, never the acceptance rule.
    first = frame.head(n//2)
    remaining = frame.iloc[n//2:].head(max(n*4,n)).copy()
    columns = [c for c in CONTROL_NAMES if experiment.base[c].nunique()>1]
    span = np.array([max(float(experiment.base[c].max()-experiment.base[c].min()),1e-9) for c in columns])
    points = remaining[columns].to_numpy(float)/span
    anchors = first[columns].to_numpy(float)/span
    distance = np.linalg.norm(points[:,None,:]-anchors[None,:,:],axis=2).min(axis=1)
    chosen = []
    proximity = remaining.target_distance.to_numpy()
    for _ in range(n-len(first)):
        score = distance / (1+20*proximity)
        i = int(np.argmax(score))
        chosen.append(i)
        distance = np.minimum(distance,np.linalg.norm(points-points[i],axis=1))
        distance[chosen] = -np.inf
    return pd.concat([first,remaining.iloc[chosen]],ignore_index=True)

def generate_multiscale(experiment, parents, n, round_number):
    rows = []
    modes = [("global",None), ("local_2pct",.02), ("local_6pct",.06), ("local_15pct",.15)]
    shear_lo, shear_hi = experiment.base["Shear rate"].agg(["min","max"])
    for index in range(n):
        name, scale = modes[index % len(modes)]
        for _ in range(100):
            source = experiment.base if scale is None else parents
            anchor = source.iloc[int(experiment.rng.integers(len(source)))][experiment.required].copy()
            row = anchor.copy()
            for c,(lo,hi) in experiment.bounds.items():
                row[c] = experiment.rng.uniform(lo,hi) if scale is None else np.clip(
                    float(row[c])+experiment.rng.normal(0,scale*(hi-lo)),lo,hi)
            for c,levels in experiment.levels.items():
                if scale is None or experiment.rng.random()<.30:
                    row[c] = experiment.rng.choice(levels)
            row["Shear rate"] = derived_shear(anchor,row["Coating Speed (m/min)"],row["Coating Gap (micron)"],row["Coating ratio (%)"])
            if not shear_lo-1e-6 <= row["Shear rate"] <= shear_hi+1e-6:
                continue
            if row["Calendaring Gap (um)"] >= row["Coating Gap (micron)"]:
                continue
            row=experiment.prepare_candidate(row)
            if row is None:continue
            key = tuple(str(row[c]) for c in experiment.required)
            if key in experiment.seen:
                continue
            experiment.seen.add(key)
            row["path_id"] = experiment.next_id
            experiment.next_id += 1
            row["search_design_mode"] = name
            row["search_round"] = round_number
            rows.append(row)
            break
        else:
            raise ValueError("Could not generate a new supported specification")
    return pd.DataFrame(rows)

class InverseDesignMixin:
    def verify(self, specs, phase, seeds):
        specs=prepare_formulation(specs)
        records = []
        particles = []
        for start in range(0, len(specs), 8):
            batch = specs.iloc[start:start + 8][self.required + ['path_id']].copy()
            draws = []
            for seed in seeds:
                key = hashlib.sha256((self.signature + str(seed) + str(self.config.verify_particles_per_seed)).encode() + pd.util.hash_pandas_object(batch, index=False).to_numpy().tobytes()).hexdigest()
                path = self.out / 'cache' / f'verify_{key}.pkl.gz'
                if path.exists():
                    p = pd.read_pickle(path)
                else:
                    p = self.simulate(batch, self.config.verify_particles_per_seed, seed)
                    p.to_pickle(path, compression='gzip')
                draws.append(p)
            p = pd.concat(draws, ignore_index=True)
            summary = self.summarise(batch, p, phase, self.config.max_tail_share_verify)
            per_seed = p.groupby(['path_id', 'verification_seed'])[[GRAV, VOL]].mean()
            stable = per_seed.apply(lambda r: bool(old.within_target(r, self.target, 0.02)), axis=1).groupby(level=0).all()
            seed_checks = []
            for seed, g in p.groupby('verification_seed'):
                per = self.summarise(batch, g, 'seed_check', self.config.max_tail_share_verify)
                seed_checks.append(per.set_index('path_id').eligible)
            summary['mean_matches_target'] = old.within_target(summary[[GRAV + '__mean', VOL + '__mean']], self.target, 0.02)
            summary['both_seed_means_match'] = summary.path_id.map(stable)
            summary['both_seed_eligibility'] = summary.path_id.map(pd.concat(seed_checks, axis=1).all(axis=1))
            summary['accepted'] = summary.eligible & summary.mean_matches_target & summary.both_seed_means_match & summary.both_seed_eligibility
            records.append(summary)
            particles.append(p)
            self._log(f'{self.method} {self.config.discharge_rate:g}C {self.out.name} {phase}: {start + len(batch)}/{len(specs)}, accepted {int(summary.accepted.sum())}', flush=True)
        return (pd.concat(records, ignore_index=True), pd.concat(particles, ignore_index=True))

    def inverse(self, min_alternatives=10, max_rounds=6):
        cfg = self.config
        if (self.out / 'inverse_complete.json').exists():
            marker=json.loads((self.out/'inverse_complete.json').read_text())
            if marker.get('signature')!=self.signature:
                raise ValueError('Saved inverse results use a different configuration or implementation. Choose a fresh output directory to preserve them.')
            self.all_paths = pd.read_csv(self.out / 'reports/searched_paths.csv')
            self.verified = pd.read_csv(self.out / 'reports/verified_alternatives.csv')
            self.alternatives = self.verified[self.verified.accepted].copy()
            self.accepted_particles = pd.read_pickle(self.out / 'reports/alternative_particles.pkl.gz')
            self.next_id = int(self.all_paths.path_id.max()) + 1
            return
        self.verified = pd.DataFrame()
        self.accepted_particles = pd.DataFrame()
        rounds = []
        self.next_id = int(self.all_paths.path_id.max()) + 1
        self.seen.update((tuple((str(row[c]) for c in self.required)) for _, row in self.all_paths.iterrows()))
        for round_number in range(max_rounds + 1):
            eligible = self.all_paths[self.all_paths.eligible]
            nearest = diverse_nearest(self, eligible, 24)
            parents = pd.concat([self.alternatives, self.target_spec, nearest], ignore_index=True).drop_duplicates('path_id')
            self.rng = np.random.default_rng(cfg.seed + 70000 + round_number + (0 if cfg.apply_capacity_caps else 1000))
            fresh = generate_multiscale(self, parents, 256 if round_number == 0 else 512, round_number)
            screened = self.evaluate(fresh, cfg.screen_particles, cfg.seed + 60000 + round_number, 'inverse_search', cfg.max_tail_share_screen)
            self.all_paths = pd.concat([self.all_paths, screened], ignore_index=True)
            self.save(self.all_paths, 'searched_paths')
            done = set() if self.verified.empty else set(self.verified.path_id)
            pool = self.all_paths[self.all_paths.eligible & ~self.all_paths.path_id.isin(done)]
            priority = diverse_nearest(self, pool, 24 if round_number == 0 else 96)
            refined = self.evaluate(priority, cfg.refine_particles, cfg.seed + 61000 + round_number, 'inverse_refinement', cfg.max_tail_share_refine)
            self.save(refined, f'inverse_refined_round_{round_number}')
            candidates = diverse_nearest(self, refined[refined.eligible], 16 if round_number == 0 else 48)
            if candidates.empty:
                continue
            verified, p = self.verify(candidates, 'inverse_verified', [cfg.seed + 74001, cfg.seed + 75001])
            self.verified = pd.concat([self.verified, verified], ignore_index=True)
            self.alternatives = self.verified[self.verified.accepted].copy()
            self.accepted_particles = pd.concat([self.accepted_particles, p[p.path_id.isin(verified[verified.accepted].path_id)]], ignore_index=True)
            self.save(self.verified, 'verified_alternatives')
            self.accepted_particles.to_pickle(self.out / 'reports/alternative_particles.pkl.gz', compression='gzip')
            rounds.append(dict(round=round_number, new_screened=len(fresh), refined=len(refined), verified=len(verified), accepted=len(self.alternatives)))
            write(self.out / 'inverse_progress.json', rounds)
            if len(self.alternatives) >= min_alternatives:
                break
        write(self.out / 'inverse_complete.json', dict(signature=self.signature,rounds=rounds, accepted=len(self.alternatives), desired_alternatives=min_alternatives, relaxed_acceptance=False))

    def feasibility_penalty(self, frame):
        penalty = np.maximum(0, 0.8 - frame.physical_pass.to_numpy()) * 4
        for c, (lo, hi) in self.ipi_support.items():
            v = frame[c + '__mean'].to_numpy()
            span = max(hi - lo, 1e-09)
            penalty += np.maximum(0, lo - v) / span + np.maximum(0, v - hi) / span
        if self.config.apply_capacity_caps:
            for c, upper in self.capacity_bounds.items():
                penalty += np.maximum(0, frame[c + '__mean'].to_numpy() / upper - 1)
        return penalty

    def broader_feasibility_search(self, screens, max_rounds=6):
        """Refine near-feasible points too: 80-particle gates are noisy estimates."""
        cfg = self.config
        self.all_paths = screens.copy()
        self.next_id = int(screens.path_id.max()) + 1
        self.seen.update((tuple((str(row[c]) for c in self.required)) for _, row in screens.iterrows()))
        refined = []
        done = set()
        rounds = []
        self.target = np.array([self.capacity_bounds[GRAV], self.capacity_bounds[VOL]])
        for round_number in range(max_rounds + 1):
            pool = self.all_paths[~self.all_paths.path_id.isin(done)].copy()
            pool['feasibility_penalty'] = self.feasibility_penalty(pool)
            eligible = pool[self.apply_eligibility(pool.copy(), cfg.max_tail_share_screen).eligible]
            front = self.front(eligible) if len(eligible) else eligible
            priority = pd.concat([front, pool.sort_values(['feasibility_penalty', 'path_id'])]).drop_duplicates('path_id').head(64)
            r = self.evaluate(priority, cfg.refine_particles, cfg.seed + 81000 + round_number, 'feasibility_refinement', cfg.max_tail_share_refine)
            refined.append(r)
            done.update(r.path_id)
            self.save(pd.concat(refined, ignore_index=True), 'feasibility_refined_paths')
            good = r[r.eligible]
            if len(good):
                runs = []
                for seed in [cfg.seed + 1001, cfg.seed + 2001]:
                    runs.append(self.simulate(good, cfg.verify_particles_per_seed, seed))
                confirmed = self.summarise(good, pd.concat(runs, ignore_index=True), 'front_confirmation', cfg.max_tail_share_verify)
                if confirmed.eligible.any():
                    self.refined = confirmed
                    self.save(confirmed, 'refined_paths')
                    final = self.front(confirmed)
                    self.save(final, 'pareto_front')
                    knee, selection = old.balanced_knee(final[[GRAV + '__mean', VOL + '__mean']])
                    chosen = final.iloc[knee]
                    self.target = chosen[[GRAV + '__mean', VOL + '__mean']].to_numpy(float)
                    self.target_spec = chosen.to_frame().T
                    write(self.out / 'selected_target.json', dict(path_id=int(chosen.path_id), selection=selection, gravimetric=float(self.target[0]), volumetric=float(self.target[1]), tolerance=0.02, objective_upper_bounds=self.capacity_bounds, capacity_caps_applied=cfg.apply_capacity_caps, lower=(self.target * 0.98).tolist(), upper=(self.target * 1.02).tolist(), caveat='Approximate sampled Pareto front; additional feasibility exploration, not a global optimum'))
                    self.save(self.all_paths, 'searched_paths')
                    write(self.out / 'feasibility_expansion.json', dict(rounds=rounds, confirmed_eligible=int(confirmed.eligible.sum()), criteria_relaxed=False))
                    return True
            rounds.append(dict(round=round_number, screened=len(self.all_paths), refined=len(r), eligible_refined=int(r.eligible.sum())))
            if round_number == max_rounds:
                break
            parents = self.all_paths.copy()
            parents['feasibility_penalty'] = self.feasibility_penalty(parents)
            parents = parents.sort_values(['feasibility_penalty', 'path_id']).head(24)
            self.rng = np.random.default_rng(cfg.seed + 82000 + round_number)
            fresh = generate_multiscale(self, parents, 512, round_number + 1)
            block = self.evaluate(fresh, cfg.screen_particles, cfg.seed + 83000 + round_number, 'feasibility_expansion', cfg.max_tail_share_screen)
            self.all_paths = pd.concat([self.all_paths, block], ignore_index=True)
            self.save(self.all_paths, 'searched_paths')
            write(self.out / 'feasibility_expansion.json', dict(rounds=rounds, criteria_relaxed=False))
        self.target = None
        write(self.out / 'no_feasible_target.json', dict(status='No confirmed feasible Pareto target found within search budget', searched=len(self.all_paths), refined=sum((len(r) for r in refined)), rounds=rounds, physical_pass_min=0.8, mean_support=self.ipi_support, criteria_relaxed=False, interpretation='Not a proof of infeasibility; no optimisation target or inverse match fabricated'))
        return False

def inverse_design(experiment,target=None,min_alternatives=10,max_rounds=6):
    """Search alternatives around a confirmed or explicit capacity pair.

    Every accepted alternative must pass the unchanged eligibility gates and both
    independent seed checks; a failed search returns its evidence, not relaxed gates.
    """
    if target is not None:
        experiment.target=np.asarray(target,float)
        if experiment.target.shape!=(2,) or not np.isfinite(experiment.target).all() or np.any(experiment.target<=0):
            raise ValueError('Target must be positive [gravimetric, volumetric] capacities')
        if experiment.all_paths.empty:
            initial=pd.concat([experiment.observed_paths(),experiment.generate(experiment.config.initial_random)],ignore_index=True)
            experiment.all_paths=experiment.evaluate(initial,experiment.config.screen_particles,experiment.config.seed,
                'inverse_initial',experiment.config.max_tail_share_screen)
        eligible=experiment.all_paths[experiment.all_paths.eligible]
        if eligible.empty:raise ValueError('No eligible designs support inverse exploration')
        experiment.target_spec=diverse_nearest(experiment,eligible,1).copy()
    if experiment.target is None:raise ValueError('Provide a target or run capacity optimisation first')
    experiment.inverse(min_alternatives=min_alternatives,max_rounds=max_rounds)
    return experiment.alternatives.copy()
