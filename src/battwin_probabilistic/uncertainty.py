"""Finite-particle uncertainty summaries; no assumption of existing tail moments."""
import numpy as np
import pandas as pd

def decompose(process,frame,particles=1000,seed=42):
    sampled=process.sample(frame,n_particles=particles,seed=seed)
    rows=[]
    for target in process.targets:
        for index,g in sampled.groupby('input_row',sort=False):
            members=g.groupby('member_id')[target]
            means=members.mean();variances=members.var(ddof=0)
            aleatoric=float(variances.mean());epistemic=float(means.var(ddof=0))
            total=float(g[target].var(ddof=0))
            np.testing.assert_allclose(aleatoric+epistemic,total,rtol=1e-10,atol=1e-10)
            rows.append(dict(input_row=index,target=target,aleatoric_std=np.sqrt(aleatoric),epistemic_std=np.sqrt(epistemic),
                total_std=np.sqrt(total),particles=particles,definition='finite-particle law of total variance; equally weighted members'))
    return pd.DataFrame(rows)

def particle_summary(particles,targets,scenario='scenario_index'):
    rows=[]
    for index,g in particles.groupby(scenario,sort=False):
        for target in targets:
            y=g[target].to_numpy(float)
            if not np.isfinite(y).all():raise ValueError('Nonfinite predictive particles')
            q=np.quantile(y,[.05,.5,.95])
            rows.append(dict(scenario_index=index,target=target,MC_mean=y.mean(),q05=q[0],median=q[1],q95=q[2],particles=len(y)))
    return pd.DataFrame(rows)
