"""Supervised permutation importance with matched random draws."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from .config import emit,SimulationConfig
from .formulation import COMPONENTS,SOLIDS,key as formulation_key,component_fields

def permutation_importance(frame,targets,predict,features=None,repeats=5,seed=42,callback=None):
    """predict(frame) returns a DataFrame containing one point forecast per target/row."""
    features=features or [c for c in frame if c not in targets]
    baseline=predict(frame).reset_index(drop=True)
    original={t:r2_score(frame[t],baseline[t]) for t in targets}
    records=[]
    for fi,feature in enumerate(features):
        for repeat in range(repeats):
            work=frame.copy()
            permutation=np.random.default_rng(seed+fi*100+repeat).permutation(len(work))
            work[feature]=work[feature].to_numpy()[permutation]
            prediction=predict(work).reset_index(drop=True)
            for target in targets:
                value=r2_score(frame[target],prediction[target])
                records.append(dict(feature=feature,target=target,repeat=repeat,baseline_R2=original[target],permuted_R2=value,importance=original[target]-value))
            emit(callback,'pfi',feature=feature,repeat=repeat+1,repeats=repeats)
    raw=pd.DataFrame(records)
    summary=raw.groupby(['target','feature'],as_index=False).agg(importance_mean=('importance','mean'),importance_std=('importance','std'),repeats=('repeat','size'))
    return raw,summary

def process_importance(process,frame,repeats=5,seed=42,callback=None):
    def predict(work):return process.predict(work).pivot(index='input_row',columns='target',values='median')
    features=list(process.features);fields=component_fields(frame)
    if len(fields)==len(COMPONENTS):
        features=[c for c in features if formulation_key(c)!=formulation_key(SOLIDS)]
        features=list(dict.fromkeys([*features,*fields.values()]))
    return permutation_importance(frame,process.targets,predict,features,repeats,seed,callback)

def chain_importance(chain,frame,repeats=5,particles=100,seed=20261014,callback=None,simulation_seed=42,batch_size=32):
    from dataclasses import replace
    from .chain import TARGETS
    config=replace(chain.config,particles=particles,seed=simulation_seed,batch_size=batch_size)
    def predict(work):
        _,draws=chain.simulate(work,config)
        return draws.groupby('scenario_index')[TARGETS].median()
    # Source measured intermediate columns are retained for fair individual comparison;
    # the chain overwrites generated states, as in the original framework.
    return permutation_importance(frame,TARGETS,predict,[c for c in frame if c not in TARGETS and not c.startswith('_')
        and c not in ['condition_id','group_id','partition','Paper'] and formulation_key(c)!=formulation_key(SOLIDS)],repeats,seed,callback)
