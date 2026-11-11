"""Calibrated inference from one combined joblib per process."""
from pathlib import Path
import numpy as np
import pandas as pd
from .artifacts import load_artifact
from .config import project_root
from .calibration import CDFMap
from .distributions import Mixture,prediction_quantiles
from .numerics import enable
from .formulation import prepare_formulation
from .promoted_inputs import density_inputs,engineered_inputs,MASS,DRY_T,DRY_P

class ProcessModel:
    def __init__(self,path,calibration='kuleshov'):
        self.artifact_path=Path(path).resolve()
        self._initialise(load_artifact(path),calibration)

    def _initialise(self,artifact,calibration):
        if calibration not in ['kuleshov','uncalibrated']:raise ValueError('Use kuleshov or uncalibrated')
        enable()
        self.calibration=calibration;self.artifact=artifact;self.stage=artifact['stage']
        self.metadata=artifact['metadata'];self.bundle=artifact['model_bundle']
        self.maps={t:CDFMap(v['x'],v['y']) for t,v in artifact['kuleshov_calibration'].items()}
        self.features=list(self.bundle['feature_columns']);self.targets=list(self.bundle['target_ensembles'])

    @classmethod
    def from_artifact(cls,artifact,calibration='kuleshov'):
        result=cls.__new__(cls);result.artifact_path=None;result._initialise(artifact,calibration);return result

    def _frame(self,frame):
        if not isinstance(frame,pd.DataFrame):raise TypeError('Provide a pandas DataFrame')
        frame=prepare_formulation(frame,require_complete=False)
        if self.bundle.get('input_recipe'):frame=density_inputs(frame)
        missing=[c for c in self.features if c not in frame]
        if missing:raise ValueError(f'{self.stage} missing exact feature columns: {missing}. See configs/processes/ and examples/data/.')
        if not len(frame):raise ValueError('Provide at least one input row')
        return frame[self.features]

    def target_frame(self,frame,target,generated=False):
        supplied=frame
        frame=self._frame(frame)
        if target not in self.targets:raise ValueError(f'Choose a target from {self.targets}')
        if not self.bundle.get('input_recipe'):return frame
        if generated and self.stage=='coating_drying' and target==DRY_P:
            frame=frame.copy()
            for parent in [MASS,DRY_T]:frame[parent]=supplied[parent].to_numpy()
        if self.stage=='coating_drying' and target==DRY_P and not generated:
            # Point-input predictions never use measured targets passed with X.
            frame=frame.copy()
            nodes,weights=np.polynomial.legendre.leggauss(61)
            levels=(nodes+1)/2
            for parent in [MASS,DRY_T]:
                mix=self._mixture(frame,parent)
                q=prediction_quantiles(mix,levels,'Kuleshov' if self.calibration=='kuleshov' else 'Uncalibrated',mapper=self.maps[parent])
                frame[parent]=q@(weights/2)
        engineered=engineered_inputs(frame,self.stage)
        columns=self.bundle['target_input_features'][target]
        return engineered[columns]

    def _mixture(self,frame,target,generated=False):
        return Mixture.predict(self.bundle['target_ensembles'][target],self.target_frame(frame,target,generated))

    def quantiles(self,frame,levels=(.05,.5,.95)):
        """Dict target -> (n_rows, n_levels), in original output units."""
        levels=np.atleast_1d(np.asarray(levels,float))
        if not len(levels) or not np.isfinite(levels).all() or np.any((levels<=0)|(levels>=1)):
            raise ValueError('Quantile probabilities must lie strictly between 0 and 1')
        result={}
        joint=None
        for t in self.targets:
            if self.bundle.get('input_recipe') and self.stage=='coating_drying' and t==DRY_P:
                if joint is None:joint=self.sample(frame,n_particles=1000,seed=42)
                result[t]=np.quantile(joint[t].to_numpy().reshape(len(frame),1000),levels,axis=1).T
            else:result[t]=prediction_quantiles(self._mixture(frame,t),levels,'Kuleshov' if self.calibration=='kuleshov' else 'Uncalibrated',mapper=self.maps[t])
        return result

    def predict(self,frame):
        """Long table: input_row, target, q05, median, q95. No mean-preservation assumption."""
        result=[]
        for target,q in self.quantiles(frame).items():
            result.extend(dict(input_row=i,target=target,q05=v[0],median=v[1],q95=v[2]) for i,v in enumerate(q))
        return pd.DataFrame(result)

    def cdf(self,frame,target,values):
        """Calibrated P(Y <= value). Values may be scalar or one per input row."""
        values=np.asarray(values,float)
        if values.ndim==0:values=np.full(len(frame),float(values))
        if values.shape!=(len(frame),):raise ValueError('Supply a scalar or one value per input row')
        if self.bundle.get('input_recipe') and self.stage=='coating_drying' and target==DRY_P:
            draws=self.sample(frame,n_particles=1000,seed=42)[target].to_numpy().reshape(len(frame),1000)
            return np.mean(draws<=values[:,None],axis=1)
        mix=self._mixture(frame,target)
        return self.maps[target](mix.cdf(values)) if self.calibration=='kuleshov' else mix.cdf(values)

    def sample(self,frame,n_particles=400,seed=42):
        """Paired multi-target particles. Shared member ID; target-specific residual draws.

        The map is applied to the ENSEMBLE CDF, never to individual member CDFs.
        This retains the existing ensemble-induced dependence, not a fitted residual copula.
        """
        clean=self._frame(frame)
        if not isinstance(n_particles,(int,np.integer)) or n_particles<20 or n_particles%20:
            raise ValueError('n_particles must be a positive multiple of 20')
        n=len(frame);rows=np.repeat(np.arange(n),n_particles)
        ids=np.tile(np.arange(n_particles)%20,n)
        rng=np.random.default_rng(seed)
        output=pd.DataFrame(dict(input_row=rows,particle_id=np.tile(np.arange(n_particles),n),member_id=ids))
        working=clean.iloc[rows].reset_index(drop=True)
        for target in self.targets:
            if self.bundle.get('input_recipe'):
                features=self.target_frame(working,target,generated=True)
                unique=features.drop_duplicates()
                lookup=pd.MultiIndex.from_frame(unique).get_indexer(pd.MultiIndex.from_frame(features))
                mix=Mixture.predict(self.bundle['target_ensembles'][target],unique).subset(lookup)
            else:mix=self._mixture(frame,target).subset(rows)
            latent=mix.sample_component_latent(rng.uniform(1e-12,1-1e-12,len(rows)),ids)
            z=latent
            if self.calibration=='kuleshov':
                ranks=mix.cdf_z(latent)
                z=mix.quantile_z(self.maps[target].inverse(ranks))[:,0]
            output[target]=mix.to_original(z)
            working[target]=output[target].to_numpy()
        return output

class ModelCatalog:
    def __init__(self,models_dir=None,project=None):
        self.models_dir=Path(models_dir) if models_dir else project_root(project)/'artifacts/models'
        self._models={}

    def load(self,stage,calibration='kuleshov'):
        key=(stage,calibration)
        if key not in self._models:self._models[key]=ProcessModel(self.models_dir/f'{stage}.joblib',calibration)
        return self._models[key]

    def registry(self):
        rows=[]
        for stage in ['mixing','coating_drying','calendaring','electrochemical']:
            model=self.load(stage)
            for target,ensemble in model.bundle['target_ensembles'].items():
                rows.append(dict(stage=stage,target=target,distribution=ensemble['distribution'],transformation=ensemble['target_mode'],
                    ensemble_members=len(ensemble['members']),bundle_path=str(model.artifact_path),calibration='held-out Kuleshov CDF'))
        return pd.DataFrame(rows)

def load_process(stage_or_path,project=None,calibration='kuleshov'):
    path=Path(stage_or_path)
    if path.suffix=='.joblib':return ProcessModel(path,calibration)
    return ModelCatalog(project=project).load(str(stage_or_path),calibration)
