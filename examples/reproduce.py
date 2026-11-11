"""Evaluate the supplied process models and chain on fixed data partitions."""
from pathlib import Path
import argparse,json,re,sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
from battwin_probabilistic import Project,SimulationConfig,reporting
from battwin_probabilistic.artifacts import sha256
from battwin_probabilistic.distributions import prediction_quantiles
from battwin_probabilistic.metrics import scores

STAGES=['mixing','coating_drying','calendaring','electrochemical']
LEVELS=np.array([.05,.5,.95])

def slug(value):return re.sub(r'[^A-Za-z0-9]+','_',value).strip('_')

def frame_for(project,stage,partition,rows):
    frame=project.data(stage)[partition].copy()
    if partition=='development':frame=frame.sort_values('_source_row',kind='stable')
    if rows is not None:frame=frame.iloc[:rows]
    return frame.reset_index(drop=True)

def metrics(observed,mean,q,context,definition):
    common=dict(q05=q[:,0],q50=q[:,1],q95=q[:,2])
    return [dict(**context,point_definition=definition,**scores(observed,dict(mean=mean,**common))),
            dict(**context,point_definition='median',**scores(observed,dict(mean=q[:,1],**common)))]

def plots(frame,folder,target,observed,q,enabled):
    if not enabled:return
    import matplotlib.pyplot as plt
    path=folder/'figures'/slug(target)
    path.mkdir(parents=True,exist_ok=True)
    log=target=='viscosity'
    reporting.parity(observed,q[:,1],target,log=log,path=path/'parity')
    reporting.uncertainty(observed,q[:,1],q[:,0],q[:,2],target,log=log,path=path/'uncertainty')
    plt.close('all')

def evaluate_process(project,stage,frame,method,folder,args):
    model=project.model(stage,method);folder.mkdir(parents=True,exist_ok=True)
    records=[];rows=[];joint=None
    nodes,weights=np.polynomial.legendre.leggauss(61)
    integration_levels=(nodes+1)/2;integration_weights=weights/2
    for target in model.targets:
        conditional=stage=='coating_drying' and target=='Dried Porosity (%)' and model.bundle.get('input_recipe')
        if conditional:
            if joint is None:joint=model.sample(frame,n_particles=args.particles,seed=args.seed)
            draws=joint[target].to_numpy().reshape(len(frame),args.particles)
            q=np.quantile(draws,LEVELS,axis=1).T;mean=draws.mean(axis=1)
            definition='finite_particle_mean'
        else:
            mixture=model._mixture(frame,target)
            label='Kuleshov' if method=='kuleshov' else 'Uncalibrated'
            q=prediction_quantiles(mixture,LEVELS,label,mapper=model.maps[target])
            mean=prediction_quantiles(mixture,integration_levels,label,mapper=model.maps[target])@integration_weights
            definition='quadrature_mean_61'
        observed=frame[target].to_numpy(float)
        context=dict(scope=stage,partition=args.partition,calibration=method,target=target)
        rows.extend(metrics(observed,mean,q,context,definition))
        records.extend(dict(**context,input_row=i,source_row=int(frame.loc[i,'_source_row']),observed=observed[i],
            mean=mean[i],median=q[i,1],q05=q[i,0],q95=q[i,2],mean_definition=definition) for i in range(len(frame)))
        plots(frame,folder,target,observed,q,not args.no_figures)
    pd.DataFrame(records).to_csv(folder/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    return rows

def evaluate_chain(project,frame,method,folder,args):
    folder.mkdir(parents=True,exist_ok=True)
    config=SimulationConfig(particles=args.particles,seed=args.seed,batch_size=args.batch_size,calibration=method,physical_porosity=False)
    summary,particles=project.chain(config).simulate(frame)
    summary.to_csv(folder/'summary.csv',index=False)
    particles.to_pickle(folder/'particles.pkl.gz',compression='gzip')
    available=[t for stage in STAGES for t in project.schema(stage)['targets'] if t in frame and t in particles]
    ordered=particles.sort_values(['scenario_index','particle_id'],kind='stable')
    records=[];rows=[]
    for target in available:
        draws=ordered[target].to_numpy(float).reshape(len(frame),args.particles)
        if not np.isfinite(draws).all():raise ValueError('Nonfinite particles for '+target+'; no particles were discarded.')
        q=np.quantile(draws,LEVELS,axis=1).T;mean=draws.mean(axis=1)
        observed=frame[target].to_numpy(float)
        context=dict(scope='chain',partition=args.partition,calibration=method,target=target)
        rows.extend(metrics(observed,mean,q,context,'finite_particle_mean'))
        records.extend(dict(**context,input_row=i,source_row=int(frame.loc[i,'_source_row']),observed=observed[i],
            mean=mean[i],median=q[i,1],q05=q[i,0],q95=q[i,2],mean_definition='finite_particle_mean') for i in range(len(frame)))
        plots(frame,folder,target,observed,q,not args.no_figures)
    pd.DataFrame(records).to_csv(folder/'predictions.csv',index=False)
    pd.DataFrame(rows).to_csv(folder/'metrics.csv',index=False)
    return rows

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--partition',choices=['train','validation','test','development'],default='test')
    parser.add_argument('--particles',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--batch-size',type=int,default=16)
    parser.add_argument('--rows',type=int,help='Limit rows for a smoke check; omit for full evaluation.')
    parser.add_argument('--stages',nargs='+',choices=STAGES,default=STAGES)
    parser.add_argument('--skip-chain',action='store_true')
    parser.add_argument('--no-figures',action='store_true')
    args=parser.parse_args()
    if args.particles<20 or args.particles%20:parser.error('Particles must be a positive multiple of 20.')
    if args.batch_size<1 or args.rows is not None and args.rows<2:parser.error('Use batch size >= 1 and at least two rows.')
    import torch
    torch.set_num_threads(1)
    project=Project(ROOT);output=args.output.resolve()
    if output==ROOT or any(output.is_relative_to(ROOT/p) for p in ['src','data','artifacts','configs','tests','docs','examples']):
        parser.error('Choose a separate output directory, such as runs/reproduction.')
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    for stage in args.stages:
        frame=frame_for(project,stage,args.partition,args.rows)
        for method in ['uncalibrated','kuleshov']:
            print(f'Evaluating {stage}, {args.partition}, {method}: {len(frame)} rows',flush=True)
            rows.extend(evaluate_process(project,stage,frame,method,output/'individual'/stage/method,args))
    if not args.skip_chain:
        frame=frame_for(project,'electrochemical',args.partition,args.rows)
        for method in ['uncalibrated','kuleshov']:
            print(f'Evaluating chain, {args.partition}, {method}: {len(frame)} rows',flush=True)
            rows.extend(evaluate_chain(project,frame,method,output/'chain'/method,args))
    pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
    settings=dict(partition=args.partition,rows_limit=args.rows,particles=args.particles,seed=args.seed,
        batch_size=args.batch_size,physical_porosity=False,stages=args.stages,
        calibration_modes=['uncalibrated','kuleshov'],model_sha256={s:sha256(ROOT/'artifacts/models'/(s+'.joblib')) for s in STAGES},
        data_sha256={str(p.relative_to(ROOT)).replace('\\','/'):sha256(p) for p in sorted((ROOT/'data/stages').glob('*/*.csv'))})
    (output/'settings.json').write_text(json.dumps(settings,indent=2)+'\n',encoding='utf-8')
    print('Saved predictions, metrics and settings to '+str(output),flush=True)

if __name__=='__main__':main()
