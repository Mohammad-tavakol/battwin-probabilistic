"""Command-line adapter over the same public APIs intended for a future GUI."""
from pathlib import Path
import argparse,json
import pandas as pd
from .project import Project
from .config import SimulationConfig,TrainingConfig,TuningConfig
from .data import read_table

def main(argv=None):
    parser=argparse.ArgumentParser(prog='battwin',description='Probabilistic battery process modelling and design')
    parser.add_argument('--project',type=Path,help='Project workspace containing configs/, data/ and artifacts/')
    commands=parser.add_subparsers(dest='command',required=True)
    commands.add_parser('inspect',help='List frozen processes, targets and distributions')
    predict=commands.add_parser('predict',help='Calibrated individual-process predictions')
    predict.add_argument('stage');predict.add_argument('input',type=Path);predict.add_argument('--output',type=Path,required=True)
    predict.add_argument('--raw',action='store_true')
    simulation=commands.add_parser('simulate',help='Propagate a fixed particle population through all processes')
    simulation.add_argument('input',type=Path);simulation.add_argument('--output',type=Path,required=True)
    simulation.add_argument('--particles',type=int,default=1000);simulation.add_argument('--seed',type=int,default=42)
    simulation.add_argument('--batch-size',type=int,default=16);simulation.add_argument('--raw',action='store_true')
    simulation.add_argument('--direct-porosity',action='store_true')
    simulation.add_argument('--physical-porosity',action='store_true',help='Explicit equation-based porosity diagnostic')
    training=commands.add_parser('train',help='Train a new artifact; never overwrite the frozen models implicitly')
    training.add_argument('stage');training.add_argument('--output',type=Path,required=True)
    training.add_argument('--tune',action='store_true');training.add_argument('--trials',type=int,default=50)
    cv=commands.add_parser('cv',help='Grouped CV with the saved configuration fixed')
    cv.add_argument('stage');cv.add_argument('--folds',type=int,default=5);cv.add_argument('--output',type=Path,required=True)
    calibration=commands.add_parser('calibrate',help='Fit CDF maps using an explicit held-out validation table')
    calibration.add_argument('stage');calibration.add_argument('model',type=Path);calibration.add_argument('validation',type=Path)
    calibration.add_argument('--output',type=Path,required=True)
    for name in ['optimise','inverse']:
        design=commands.add_parser(name,help='Capacity design search using development data only')
        design.add_argument('--rate',type=float,default=.2);design.add_argument('--chemistry',default='NMC622')
        design.add_argument('--uncapped',action='store_true');design.add_argument('--raw',action='store_true')
        design.add_argument('--config',type=Path);design.add_argument('--output',type=Path,required=True)
        if name=='inverse':design.add_argument('--target',type=float,nargs=2,required=True,metavar=('GRAV','VOL'))
    scrap=commands.add_parser('scrap',help='First-stage rejection and flag-only final disposition')
    scrap.add_argument('particles',type=Path);scrap.add_argument('--target',type=float,nargs=2,required=True)
    scrap.add_argument('--tolerances',type=float,nargs='+',default=[2,3,5,10]);scrap.add_argument('--output',type=Path,required=True)
    pfi=commands.add_parser('pfi',help='Supervised permutation importance, training set by default')
    pfi.add_argument('scope',choices=['mixing','coating_drying','calendaring','electrochemical','chain'])
    pfi.add_argument('--repeats',type=int,default=5);pfi.add_argument('--particles',type=int,default=100)
    pfi.add_argument('--output',type=Path,required=True)
    oot=commands.add_parser('validate-repetitions',help='Experimental-reference OOT, not manufacturing scrap')
    oot.add_argument('--raw-particles',type=Path,required=True);oot.add_argument('--calibrated-particles',type=Path,required=True)
    oot.add_argument('--reference',choices=['experimental_mean','leave_one_out','leave_one_out_pairs'],default='experimental_mean')
    oot.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv);project=Project(args.project)
    def callback(event):print(json.dumps(event,default=str),flush=True)
    if args.command=='inspect':print(project.catalog.registry().to_string(index=False));return
    if args.command=='predict':
        result=project.model(args.stage,'uncalibrated' if args.raw else 'kuleshov').predict(read_table(args.input))
        args.output.parent.mkdir(parents=True,exist_ok=True);result.to_csv(args.output,index=False);return
    args.output.mkdir(parents=True,exist_ok=True) if args.command!='calibrate' else args.output.parent.mkdir(parents=True,exist_ok=True)
    if args.command=='simulate':
        cfg=SimulationConfig(args.particles,args.seed,args.batch_size,'uncalibrated' if args.raw else 'kuleshov',args.physical_porosity and not args.direct_porosity)
        summary,particles=project.chain(cfg).simulate(read_table(args.input),callback=callback)
        summary.to_csv(args.output/'summary.csv',index=False);particles.to_pickle(args.output/'particles.pkl.gz',compression='gzip')
        return
    if args.command=='train':
        project.train(args.stage,args.output,tuning=TuningConfig(trials=args.trials),use_selected=not args.tune,callback=callback);return
    if args.command=='cv':
        from .training import cross_validate
        data=project.data(args.stage)
        detail,summary=cross_validate(data['development'],data['schema'],data['schema']['target_configurations'],TrainingConfig(cv_folds=args.folds),callback)
        detail.to_csv(args.output/'fold_metrics.csv',index=False);summary.to_csv(args.output/'mean_std.csv');return
    if args.command=='calibrate':
        from .artifacts import load_artifact,save_artifact
        from .training import recalibrate
        save_artifact(recalibrate(load_artifact(args.model),read_table(args.validation),project.schema(args.stage)),args.output);return
    if args.command in ['optimise','inverse']:
        from .design_space import SearchConfig
        from .optimisation import optimise_capacity,Experiment
        from .inverse_design import inverse_design
        values=json.loads(args.config.read_text()) if args.config else {}
        values.update(chemistry=args.chemistry,discharge_rate=args.rate,apply_capacity_caps=not args.uncapped)
        cfg=SearchConfig(**values);method='uncalibrated' if args.raw else 'kuleshov'
        development=project.data('electrochemical')['development']
        if args.command=='optimise':optimise_capacity(development,cfg,project.catalog,args.output,method,callback)
        else:
            experiment=Experiment(development,cfg,project.catalog,args.output,method,callback)
            inverse_design(experiment,target=args.target)
        return
    if args.command=='scrap':
        from .scrap import assess_scrap
        assess_scrap(pd.read_pickle(args.particles),args.target,[(t/100,t/100) for t in args.tolerances],args.output);return
    if args.command=='pfi':
        from .pfi import chain_importance,process_importance
        if args.scope=='chain':raw,summary=chain_importance(project.chain(),project.data('electrochemical')['train'],args.repeats,args.particles,callback=callback)
        else:raw,summary=process_importance(project.model(args.scope),project.data(args.scope)['train'],args.repeats,callback=callback)
        raw.to_csv(args.output/'repeats.csv',index=False);summary.to_csv(args.output/'summary.csv',index=False);return
    if args.command=='validate-repetitions':
        from .validation import experimental_oot
        data=project.data('electrochemical');features=data['schema']['split_group_features']
        result=experimental_oot(data['test'],{'Full chain':{'Uncalibrated':pd.read_pickle(args.raw_particles),
            'Kuleshov':pd.read_pickle(args.calibrated_particles)}},features,[c for c in features if c!='Discharge rate (*C)'],
            data['schema']['targets'],reference=args.reference)
        for name,table in result.items():table.to_csv(args.output/(name+'.csv'),index=False)
