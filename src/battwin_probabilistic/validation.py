"""Experimental-reference OOT validation, separate from manufacturing scrap."""
import hashlib
import numpy as np
import pandas as pd
from .data import condition_id

def experimental_oot(observations,predictions,condition_features,manufacturing_features,targets,
                     tolerances=(2,3,5,10),reference='experimental_mean',bootstrap=5000,seed=20261015):
    """Predictions: dict(scope -> dict(method -> paired particle DataFrame)).

    Particle rows need scenario_index matching the observation position. Both targets
    remain in the same particle row. Singleton conditions are excluded. LOO triplets
    and pair sensitivity are separate references: leave_one_out / leave_one_out_pairs.
    """
    if reference not in ['experimental_mean','leave_one_out','leave_one_out_pairs']:raise ValueError('Unknown reference')
    if len(targets)!=2:raise ValueError('Joint electrochemical validation requires two paired targets')
    frame=observations.reset_index(drop=True).copy()
    frame['condition_id']=condition_id(frame,condition_features)
    frame['manufacturing_id']=condition_id(frame,manufacturing_features)
    rows=[];refs=[];eligibility=[]
    for label,g in frame.groupby('condition_id',sort=False):
        n=len(g)
        eligible=n>=2 if reference=='experimental_mean' else n>=3 if reference=='leave_one_out' else n==2
        eligibility.append(dict(condition_id=label,n_repetitions=n,eligible=eligible,reference=reference))
        if not eligible:continue
        y=g[targets].to_numpy(float);mean=y.mean(axis=0)
        if np.any(mean<=0):raise ValueError('Relative OOT requires positive experimental references')
        for j,index in enumerate(g.index):
            center=mean if reference=='experimental_mean' else (y.sum(axis=0)-y[j])/(n-1)
            for k,target in enumerate(targets):refs.append(dict(condition_id=label,source_row=index,target=target,actual=y[j,k],reference=center[k]))
            for tol in tolerances:
                lo=center*(1-tol/100);hi=center*(1+tol/100)
                observed=(y[j]<lo)|(y[j]>hi)
                for scope,methods in predictions.items():
                    for method,particles in methods.items():
                        d=particles[particles.scenario_index.isin(g.index)][targets].to_numpy(float)
                        if not len(d):raise ValueError('Missing paired particles')
                        rejected=(d<lo)|(d>hi)
                        for k,event in enumerate([*targets,'Joint']):
                            rows.append(dict(reference=reference,scope=scope,method=method,event=event,tolerance_pct=tol,
                                condition_id=label,manufacturing_id=g.manufacturing_id.iloc[0],source_row=index,n_repetitions=n,
                                Chemistry=g.Chemistry.iloc[0] if 'Chemistry' in g else '',
                                rate_C=float(g['Discharge rate (*C)'].iloc[0]) if 'Discharge rate (*C)' in g else np.nan,
                                observed_oot=int(observed.any() if k==2 else observed[k]),
                                predicted_oot=float(rejected.any(axis=1).mean() if k==2 else rejected[:,k].mean()),
                                reference_1=center[0],reference_2=center[1],lower_1=lo[0],upper_1=hi[0],lower_2=lo[1],upper_2=hi[1]))
    raw=pd.DataFrame(rows)
    if raw.empty:return dict(rows=raw,conditions=pd.DataFrame(),summary=pd.DataFrame(),references=pd.DataFrame(refs),eligibility=pd.DataFrame(eligibility),reliability=pd.DataFrame())
    keys=['reference','scope','method','event','tolerance_pct','condition_id','manufacturing_id','n_repetitions','Chemistry','rate_C']
    conditions=raw.groupby(keys,dropna=False,as_index=False)[['observed_oot','predicted_oot']].mean()
    summaries=[];reliability=[]
    cohorts={'All eligible':np.ones(len(conditions),bool)}
    for rate in [.2,1.]:cohorts[f'NMC622 {rate:g}C']=conditions.Chemistry.eq('NMC622')&np.isclose(conditions.rate_C,rate)
    for cohort,mask in cohorts.items():
        for key,g in conditions[mask].groupby(['scope','event','tolerance_pct'],sort=False):
            for method,d in g.groupby('method',sort=False):
                groups=np.unique(d.manufacturing_id)
                counts=np.array([d.manufacturing_id.eq(c).sum() for c in groups])
                sums=np.array([d.loc[d.manufacturing_id.eq(c),['observed_oot','predicted_oot']].sum().to_numpy() for c in groups])
                keyseed=seed+int(hashlib.sha256(f'{reference}|{cohort}|{key}'.encode()).hexdigest()[:8],16)
                choice=np.random.default_rng(keyseed).integers(len(groups),size=(bootstrap,len(groups)))
                values=sums[choice].sum(axis=1)/counts[choice].sum(axis=1)[:,None]
                ci=np.quantile(values,[.025,.975],axis=0)
                summaries.append(dict(reference=reference,cohort=cohort,scope=key[0],method=method,event=key[1],tolerance_pct=key[2],
                    n_conditions=len(d),n_observations=int(d.n_repetitions.sum()),n_manufacturing_groups=len(groups),
                    observed_oot_pct=100*d.observed_oot.mean(),predicted_oot_pct=100*d.predicted_oot.mean(),
                    observed_low_pct=100*ci[0,0],observed_high_pct=100*ci[1,0],predicted_low_pct=100*ci[0,1],predicted_high_pct=100*ci[1,1]))
    for key,g in conditions.groupby(['scope','method','event','tolerance_pct'],sort=False):
        binid=np.minimum((g.predicted_oot*3).astype(int),2)
        for b in range(3):
            d=g[binid.eq(b)]
            reliability.append(dict(scope=key[0],method=key[1],event=key[2],tolerance_pct=key[3],probability_bin=b,
                n_conditions=len(d),n_manufacturing_groups=d.manufacturing_id.nunique(),supported=len(d)>=5 and d.manufacturing_id.nunique()>=3,
                predicted_oot_pct=100*d.predicted_oot.mean(),observed_oot_pct=100*d.observed_oot.mean()))
    return dict(rows=raw,conditions=conditions,summary=pd.DataFrame(summaries),references=pd.DataFrame(refs),
        eligibility=pd.DataFrame(eligibility),reliability=pd.DataFrame(reliability))
