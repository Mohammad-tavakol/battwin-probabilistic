"""Named-state orchestration with fixed-particle propagation."""
from pathlib import Path
import numpy as np
import pandas as pd
from .config import SimulationConfig,emit
from .models import ModelCatalog
from .distributions import Mixture
from .formulation import prepare_formulation
from .promoted_inputs import density_inputs
from .orchestration.core import ProbabilisticStage,ProbabilisticNamedStateSpaceLine,SAMPLE_QUANTILE_GRID,POROSITY_CLIP_BOUNDS

STAGES=(('mixing','Mixing'),('coating_drying','Drying'),('calendaring','Calendaring'),('electrochemical','Electrochemical'))
TARGETS=['Gravimetric capacity (mA h/g)','Volumetric capacity (mA h/cm3)']

class CalibratedStage(ProbabilisticStage):
    def __init__(self,process,stage_name,physical_specs,**kwargs):
        self.process=process
        self.physical_porosity_specs=physical_specs
        self.use_physical_porosity=bool(physical_specs)
        super().__init__(name=stage_name,bundle_path=str(process.artifact_path),bundle=process.bundle,
            apply_bundle_calibration=False,calibration_label=process.calibration,**kwargs)

    def physical_spec(self,output):
        return self.physical_porosity_specs.get(output)

    def sample_target(self,target,particles,member_ids,rng,**kwargs):
        original=self.process.target_frame(particles,target,generated=True)
        unique=original.drop_duplicates()
        lookup=pd.MultiIndex.from_frame(unique).get_indexer(pd.MultiIndex.from_frame(original))
        if np.any(lookup<0):raise ValueError('Failed to align particle inputs')
        mix=Mixture.predict(self.bundle['target_ensembles'][target],unique).subset(lookup)
        u=rng.uniform(1e-12,1-1e-12,len(particles))
        ids=np.asarray(member_ids,int)%len(self.bundle['target_ensembles'][target]['members'])
        z=mix.sample_component_latent(u,ids)
        if self.process.calibration=='kuleshov':
            z=mix.quantile_z(self.process.maps[target].inverse(mix.cdf_z(z)))[:,0]
        return mix.to_original(z)

    def _bundle_features_for_target(self,target):
        # Raw stage inputs are checked by orchestration; derived target inputs
        # are constructed only after the necessary upstream outputs exist.
        return list(self.bundle.get('target_input_features',{}).get(target,self.bundle['feature_columns']))

class ProcessChain(ProbabilisticNamedStateSpaceLine):
    def __init__(self,catalog=None,config=None):
        self.catalog=catalog or ModelCatalog()
        self.config=config or SimulationConfig()
        processes={key:self.catalog.load(key,self.config.calibration) for key,_ in STAGES}
        canonical=processes['electrochemical'].features+processes['electrochemical'].targets
        lookup={c.casefold():c for c in canonical}
        stages=[]
        for key,name in STAGES:
            process=processes[key]
            aliases={c:lookup[c.casefold()] for c in process.features if c.casefold() in lookup and c!=lookup[c.casefold()]}
            output_aliases={process.targets[0]:['Viscosity (Pa.s)']} if key=='mixing' else {}
            physical={}
            if self.config.physical_porosity and key in ['coating_drying','calendaring']:
                porosity='Dried Porosity (%)' if key=='coating_drying' else 'Calendered porosity (%)'
                thickness=process.targets[1] if key=='coating_drying' else 'Calendered thickness (um)'
                physical={porosity:dict(mass_loading_col='Mass loading dry (mg/cm2)',thickness_col=thickness,
                    bulk_density_col='Solid density reference (g/cm3)',scale=10.,clip_bounds=POROSITY_CLIP_BOUNDS)}
            stages.append(CalibratedStage(process,name,physical,input_columns=None,output_columns=process.targets,
                output_aliases=output_aliases,input_aliases=aliases,required_input_columns=[aliases.get(c,c) for c in process.features]))
        super().__init__(stages);self.compile()

    def align_inputs(self,frame):
        result=prepare_formulation(frame)
        types=result['Electrode type'].astype(str).str.strip().str.strip("'\"").str.casefold()
        result['Solid density reference (g/cm3)']=types.map({'cathode':4.447,'anode':2.158})
        if result['Solid density reference (g/cm3)'].isna().any():raise ValueError('Unknown electrode type for solid density')
        if any(p.process.bundle.get('input_recipe') for p in self.stages):result=density_inputs(result)
        return result

    def simulate_items_particles(self,df_input,*args,**kwargs):
        return super().simulate_items_particles(self.align_inputs(df_input),*args,**kwargs)

    def get_required_input_columns(self):
        required=set();generated=set()
        for stage in self.stages:
            required.update(set(stage.required_input_columns)-generated)
            local=set(stage.output_columns)
            for output in stage.output_columns:
                spec=stage.physical_spec(output)
                if spec:required.update({spec['mass_loading_col'],spec['thickness_col'],spec['bulk_density_col']}-generated-local)
            generated.update(stage.get_generated_output_columns(include_model_comparison=True))
            for aliases in stage.output_aliases.values():generated.update(aliases)
        return sorted(required)

    def simulate(self,frame,config=None,callback=None):
        cfg=config or self.config
        if cfg.particles<20 or cfg.particles%20:raise ValueError('Particles must be a positive multiple of 20')
        if cfg.batch_size<1 or not len(frame):raise ValueError('Nonempty inputs and a positive batch size are required')
        if cfg.calibration!=self.config.calibration or cfg.physical_porosity!=self.config.physical_porosity:
            raise ValueError('Build a chain with the requested calibration/porosity settings')
        summaries=[];particles=[]
        for start in range(0,len(frame),cfg.batch_size):
            batch=frame.iloc[start:start+cfg.batch_size].reset_index(drop=True)
            summary,p=self.simulate_items_particles(batch,n_particles_override=cfg.particles,
                n_member_repeats=cfg.particles//20,random_state=cfg.seed+start,final_kpi_columns=TARGETS,
                spec_limits={},quantile_grid=SAMPLE_QUANTILE_GRID,n_quantile_samples=1000,keep_particles=True)
            for table in [summary,p]:
                for col in ['scenario_index','scenario_id']:
                    if col in table:table[col]+=start
            summaries.append(summary);particles.append(p)
            emit(callback,'simulation_batch',completed=min(start+cfg.batch_size,len(frame)),total=len(frame),particles=cfg.particles)
        return pd.concat(summaries,ignore_index=True),pd.concat(particles,ignore_index=True)
