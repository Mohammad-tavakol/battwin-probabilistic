"""Development-only screening and Optuna TPE tuning of the top alternatives."""
import copy,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import optuna
from .ensemble import build_preprocessor,to_dense_if_needed,transform_y,normalise_xgb_params,make_lss_model,predict_model_quantile_grid
from ..metrics import scores
from ..data import grouped_split,annotate
from ..config import TuningConfig,emit
import xgboost as xgb

GRID=np.round(np.linspace(.01,.99,99),4)

def fit_single(frame,schema,target,config,seed=42):
    import torch
    torch.set_num_threads(1)
    pp,_,_=build_preprocessor(frame[schema['features']],schema['categorical_features'])
    x=to_dense_if_needed(pp.fit_transform(frame[schema['features']]))
    y=transform_y(frame[target].to_numpy(float),config['target_mode'])
    params=normalise_xgb_params(config['xgb_params'],seed)
    params['nthread']=1
    model=make_lss_model(config['distribution'],config['dist_kwargs'])
    model.train(params,xgb.DMatrix(x,label=y),num_boost_round=config['num_boost_round'])
    return model,pp

def candidate_scores(model,pp,frame,schema,target,config,seed):
    grid=predict_model_quantile_grid(model,pp,frame[schema['features']],config['target_mode'],GRID,1000,seed)
    return scores(frame[target],dict(mean=grid.mean(axis=0),q05=np.quantile(grid,.05,axis=0),q50=np.quantile(grid,.5,axis=0),q95=np.quantile(grid,.95,axis=0)))

def select_configurations(training,schema,tuning=None,selection_fraction=.2,output_dir=None,callback=None):
    if 'partition' in training and training.partition.astype(str).str.lower().eq('test').any():
        raise ValueError('Test observations cannot enter Optuna selection')
    training=annotate(training,schema)
    cfg=tuning or TuningConfig()
    if cfg.trials<1 or cfg.top_candidates<1:raise ValueError('Positive trials and candidate counts required')
    if cfg.objective not in ['R2','width']:raise ValueError('Objective must be R2 or width')
    train,selection=grouped_split(training,selection_fraction,cfg.seed)
    screens=[];tuned=[];configs={}
    if output_dir:Path(output_dir).mkdir(parents=True,exist_ok=True)
    fingerprint=hashlib.sha256(pd.util.hash_pandas_object(training[schema['features']+schema['targets']],index=True).values.tobytes()).hexdigest()[:12]
    for ti,target in enumerate(schema['targets']):
        for dist in cfg.distributions:
            for mode in cfg.transformations:
                if dist in ['Gamma','LogNormal','Weibull'] and np.any(transform_y(train[target],mode)<=0):continue
                for stabilization in ['None','MAD']:
                    config=dict(distribution=dist,target_mode=mode,num_boost_round=cfg.screening_rounds,
                        dist_kwargs=dict(stabilization=stabilization,response_fn='softplus',loss_fn='nll'),
                        xgb_params=dict(eta=.05,max_depth=3,gamma=.001,min_child_weight=5.,subsample=.85,colsample_bytree=.85,
                            **{'lambda':1.,'alpha':.01},booster='gbtree',nthread=1,verbosity=0))
                    try:
                        model,pp=fit_single(train,schema,target,config,cfg.seed)
                        m=candidate_scores(model,pp,selection,schema,target,config,cfg.seed)
                        if m['status']!='complete' or not np.isfinite([m['R2'],m['Winkler90']]).all():raise ValueError('Nonfinite candidate')
                        screens.append(dict(target=target,distribution=dist,target_mode=mode,config_json=json.dumps(config),**m))
                    except Exception as exc:
                        screens.append(dict(target=target,distribution=dist,target_mode=mode,status='failed',error=repr(exc)))
                    emit(callback,'screening',target=target,distribution=dist,transformation=mode)
        ranked=pd.DataFrame(screens)
        ranked=ranked[ranked.target.eq(target)&ranked.status.eq('complete')].sort_values(['R2','Winkler90'],ascending=[False,True]).drop_duplicates(['distribution','target_mode']).head(cfg.top_candidates)
        if ranked.empty:raise RuntimeError(f'No finite preliminary models for {target}')
        alternatives=[]
        for fi,row in enumerate(ranked.itertuples()):
            initial=json.loads(row.config_json)
            def objective(trial):
                params=dict(eta=trial.suggest_float('eta',.005,.3,log=True),max_depth=trial.suggest_int('max_depth',1,8),
                    gamma=trial.suggest_float('gamma',1e-6,2.,log=True),min_child_weight=trial.suggest_float('min_child_weight',.01,100.,log=True),
                    subsample=trial.suggest_float('subsample',.6,1.),colsample_bytree=trial.suggest_float('colsample_bytree',.6,1.),
                    **{'lambda':trial.suggest_float('lambda',1e-4,20.,log=True),'alpha':trial.suggest_float('alpha',1e-5,10.,log=True)},
                    booster='gbtree',nthread=1,verbosity=0)
                config=dict(distribution=row.distribution,target_mode=row.target_mode,xgb_params=params,
                    num_boost_round=trial.suggest_int('num_boost_round',cfg.minimum_rounds,cfg.maximum_rounds,step=cfg.round_step),
                    dist_kwargs=dict(stabilization=trial.suggest_categorical('stabilization',['None','MAD']),response_fn='softplus',loss_fn='nll'))
                try:
                    model,pp=fit_single(train,schema,target,config,cfg.seed)
                    m=candidate_scores(model,pp,selection,schema,target,config,cfg.seed)
                    if m['status']!='complete' or not np.isfinite([m['R2'],m['MPIW90']]).all():raise ValueError('Nonfinite tuned model')
                    trial.set_user_attr('config',config);trial.set_user_attr('metrics',m)
                    emit(callback,'optuna_trial',target=target,trial=trial.number,R2=m['R2'],MPIW90=m['MPIW90'])
                    return float(m['R2'] if cfg.objective=='R2' else m['MPIW90'])
                except Exception as exc:
                    trial.set_user_attr('error',repr(exc));raise optuna.TrialPruned(str(exc))
            name=f'{fingerprint}_{ti}_{row.distribution}_{row.target_mode}_{cfg.objective}'
            storage='sqlite:///'+(Path(output_dir)/'optuna.sqlite3').resolve().as_posix() if output_dir else None
            study=optuna.create_study(study_name=name,storage=storage,load_if_exists=True,
                direction='maximize' if cfg.objective=='R2' else 'minimize',sampler=optuna.samplers.TPESampler(seed=cfg.seed+ti*1000+fi,multivariate=True))
            if not study.trials:
                trial_params={**initial['xgb_params'],'num_boost_round':initial['num_boost_round'],'stabilization':initial['dist_kwargs']['stabilization']}
                trial_params['num_boost_round']=int(np.clip(trial_params['num_boost_round'],cfg.minimum_rounds,cfg.maximum_rounds))
                study.enqueue_trial(trial_params)
            finished=sum(t.state.is_finished() for t in study.trials)
            study.optimize(objective,n_trials=max(0,cfg.trials-finished))
            completed=[t for t in study.trials if t.state==optuna.trial.TrialState.COMPLETE]
            if not completed:raise RuntimeError(f'All Optuna trials failed for {name}')
            best=study.best_trial
            alternatives.append(dict(config=best.user_attrs['config'],**best.user_attrs['metrics']))
            tuned.extend(dict(target=target,distribution=row.distribution,target_mode=row.target_mode,trial=t.number,
                state=t.state.name,value=t.value,params_json=json.dumps(t.params),**t.user_attrs.get('metrics',{})) for t in study.trials)
        if cfg.objective=='R2':
            maximum=max(a['R2'] for a in alternatives)
            close=[a for a in alternatives if a['R2']>=maximum-cfg.accuracy_tie]
            best=min(close,key=lambda a:(a['Winkler90'],-a['R2']))
        else:best=min(alternatives,key=lambda a:a['MPIW90'])
        configs[target]=best['config']
    if output_dir:
        pd.DataFrame(screens).to_csv(Path(output_dir)/'screening.csv',index=False)
        pd.DataFrame(tuned).to_csv(Path(output_dir)/'optuna_trials.csv',index=False)
        (Path(output_dir)/'selected_configurations.json').write_text(json.dumps(configs,indent=2),encoding='utf-8')
    return configs,dict(screening=pd.DataFrame(screens),trials=pd.DataFrame(tuned))
