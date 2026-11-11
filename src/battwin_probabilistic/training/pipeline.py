"""Training/calibration APIs. Test data are never accepted by model selection."""
import copy,json,hashlib,contextlib,io
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from ..config import TrainingConfig,TuningConfig,emit
from ..data import annotate,assert_disjoint,condition_id,grouped_split
from ..calibration import CDFMap
from ..distributions import Mixture
from ..metrics import scores
from ..artifacts import save_artifact,ARTIFACT_TYPE
from .ensemble import train_target_ensemble
from .selection import select_configurations

def require_development(frame):
    if 'partition' in frame and frame.partition.astype(str).str.lower().eq('test').any():
        raise ValueError('Test observations cannot enter model development')

def train_process(train,validation,schema,configurations,config=None,output_path=None,callback=None):
    cfg=config or TrainingConfig()
    import torch
    torch.set_num_threads(cfg.nthread)
    for frame in [train,validation]:
        if 'partition' in frame and frame.partition.astype(str).str.lower().eq('test').any():
            raise ValueError('Test observations cannot be fitted or used for calibration')
    train=annotate(train,schema);validation=annotate(validation,schema)
    assert_disjoint(train,validation)
    if cfg.members!=20:raise ValueError('The canonical workflow requires B=20 ensemble members')
    bootstrap=condition_id(train,schema.get('bootstrap_group_features',schema['split_group_features']))
    split=dict(x_trainval=train[schema['features']],y_trainval=train[schema['targets']],
        trainval_group_keys=bootstrap,categorical_columns=schema['categorical_features'])
    ensembles={};maps={}
    for target in schema['targets']:
        with contextlib.redirect_stdout(io.StringIO()):
            ensembles[target]=train_target_ensemble(split,target,copy.deepcopy(configurations[target]),cfg.members,cfg.bootstrap_mode,cfg.seed)
        mixture=Mixture.predict(ensembles[target],validation[schema['features']])
        mapper=CDFMap.fit(mixture.cdf(validation[target].to_numpy(float)))
        maps[target]={**mapper.record(),'n_rows':len(validation),'source':'explicit held-out validation frame',
            'model_refitted':True,'validation_condition_ids':sorted(validation.condition_id.unique())}
        emit(callback,'trained_target',target=target,members=cfg.members,calibration_rows=len(validation))
    bundle=dict(bundle_type='xgboostlss_bootstrap_ensemble',created_at=datetime.now(timezone.utc).isoformat(),
        feature_columns=schema['features'],target_columns=schema['targets'],categorical_columns=schema['categorical_features'],
        numerical_columns=[c for c in schema['features'] if c not in schema['categorical_features']],target_ensembles=ensembles,
        target_configs=copy.deepcopy(configurations),settings=dict(n_members=cfg.members,random_state=cfg.seed,
            bootstrap_mode=cfg.bootstrap_mode,group_columns=schema.get('bootstrap_group_features',schema['split_group_features'])))
    artifact=dict(artifact_type=ARTIFACT_TYPE,format_version=1,stage=schema['stage'],model_bundle=bundle,kuleshov_calibration=maps,
        metadata=copy.deepcopy(schema),calibration_spec=dict(method='Kuleshov CDF',source='held-out validation',legacy_spread_factors_used=False),
        provenance=dict(model_refitted=True,calibration_refitted=True,training_rows=len(train),validation_rows=len(validation),
            training_condition_ids=sorted(train.condition_id.unique()),validation_condition_ids=sorted(validation.condition_id.unique()),seed=cfg.seed))
    if output_path:save_artifact(artifact,output_path)
    return artifact

def fit_process(development,schema,training=None,tuning=None,configurations=None,output_dir=None,callback=None):
    require_development(development)
    cfg=training or TrainingConfig()
    development=annotate(development,schema)
    fit,validation=grouped_split(development,cfg.validation_fraction,cfg.seed+17)
    reports={}
    if configurations is None:
        configurations,reports=select_configurations(fit,schema,tuning or TuningConfig(),cfg.selection_fraction,
            Path(output_dir)/'selection' if output_dir else None,callback)
    path=Path(output_dir)/f'{schema["stage"]}.joblib' if output_dir else None
    artifact=train_process(fit,validation,schema,configurations,cfg,path,callback)
    return artifact,reports

def evaluate_artifact(artifact,frame,point='median'):
    from ..models import ProcessModel
    # Evaluation uses the in-memory artifact, without serialisation or a second fit.
    model=ProcessModel.from_artifact(artifact)
    rows=[]
    for target,q in model.quantiles(frame).items():
        prediction=q[:,1]
        if point=='mean':
            sampled=model.sample(frame,n_particles=1000,seed=42)
            prediction=sampled.groupby('input_row')[target].mean().to_numpy()
        elif point!='median':raise ValueError('Point must be median or mean')
        rows.append(dict(target=target,point_definition=point,**scores(frame[target],dict(mean=prediction,q05=q[:,0],q50=q[:,1],q95=q[:,2]))))
    return pd.DataFrame(rows)

def cross_validate(development,schema,configurations,training=None,callback=None):
    require_development(development)
    from dataclasses import replace
    cfg=training or TrainingConfig();frame=annotate(development,schema);rows=[]
    folds=min(cfg.cv_folds,frame.group_id.nunique())
    if folds<2:raise ValueError('Insufficient groups for CV')
    configuration_hash=hashlib.sha256(json.dumps(configurations,sort_keys=True).encode()).hexdigest()
    for fold,(tr,va) in enumerate(GroupKFold(n_splits=folds).split(frame,groups=frame.group_id),1):
        fit,cal=grouped_split(frame.iloc[tr],cfg.validation_fraction,cfg.seed+fold)
        validation=frame.iloc[va];assert_disjoint(fit,cal,validation)
        artifact=train_process(fit,cal,schema,configurations,replace(cfg,seed=cfg.seed+fold))
        score=evaluate_artifact(artifact,validation)
        score['fold']=fold;score['evaluation']='held-out CV fold';score['configuration_hash']=configuration_hash
        rows.append(score);emit(callback,'cv_fold',fold=fold,total=folds)
    detail=pd.concat(rows,ignore_index=True)
    summary=detail.groupby('target').agg({c:['mean','std'] for c in ['R2','MAE','RMSE','MAPE','MdAPE','PICP90','MPIW90','NMPIW90','WIS90']})
    return detail,summary

def repeated_splits(development,schema,configurations,seeds=(1,42,100),training=None,callback=None):
    require_development(development)
    from dataclasses import replace
    cfg=training or TrainingConfig();frame=annotate(development,schema);rows=[]
    for seed in seeds:
        pool,holdout=grouped_split(frame,cfg.outer_holdout_fraction,seed)
        train,validation=grouped_split(pool,cfg.validation_fraction,seed+17)
        assert_disjoint(train,validation,holdout)
        artifact=train_process(train,validation,schema,configurations,replace(cfg,seed=seed))
        metrics=evaluate_artifact(artifact,holdout);metrics['seed']=seed;metrics['evaluation']='outer grouped holdout'
        rows.append(metrics);emit(callback,'outer_split',seed=seed)
    return pd.concat(rows,ignore_index=True)

def recalibrate(artifact,validation,schema=None):
    schema=schema or dict(features=artifact['model_bundle']['feature_columns'],targets=artifact['model_bundle']['target_columns'],
        split_group_features=artifact['model_bundle']['feature_columns'])
    frame=annotate(validation,schema)
    forbidden=set(artifact.get('provenance',{}).get('training_condition_ids',[]))
    if forbidden&set(frame.condition_id):raise ValueError('Calibration conditions overlap training')
    if 'partition' in frame and frame.partition.astype(str).str.lower().eq('test').any():raise ValueError('Test observations cannot fit calibration')
    result=copy.deepcopy(artifact)
    for target,ensemble in result['model_bundle']['target_ensembles'].items():
        mixture=Mixture.predict(ensemble,frame[schema['features']]);mapper=CDFMap.fit(mixture.cdf(frame[target].to_numpy(float)))
        old=result['kuleshov_calibration'].get(target,{})
        result['kuleshov_calibration'][target]={**mapper.record(),'n_rows':len(frame),'source':'explicit held-out validation frame',
            **({'base_bundle_sha256':old['base_bundle_sha256']} if 'base_bundle_sha256' in old else {})}
    result.setdefault('provenance',{})['calibration_refitted']=True
    return result
