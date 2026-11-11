"""Point and interval metrics in original response units."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
METRICS=['R2', 'MAE', 'RMSE', 'MAPE', 'MdAPE', 'PICP90', 'MPIW90', 'NMPIW90', 'Winkler90', 'WIS90']
def scores(y, pred):
    y=np.asarray(y,float); mean=np.asarray(pred['mean'],float)
    median=np.asarray(pred['q50'],float); lo=np.asarray(pred['q05'],float); hi=np.asarray(pred['q95'],float)
    if not np.isfinite(y).all():raise ValueError('Nonfinite observed response')
    point_ok=bool(np.isfinite(mean).all())
    interval_ok=bool(np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all())
    median_ok=bool(np.isfinite(median).all())
    width=hi-lo
    winkler=width+20*np.maximum(lo-y,0)+20*np.maximum(y-hi,0)
    valid=np.abs(y)>1e-12
    ape=100*np.abs((y[valid]-mean[valid])/y[valid]) if point_ok else np.array([])
    return dict(n=len(y),status='complete' if point_ok and interval_ok and median_ok else 'nonfinite_prediction',
        invalid_mean_n=int((~np.isfinite(mean)).sum()),invalid_interval_n=int((~(np.isfinite(lo)&np.isfinite(hi))).sum()),
        R2=r2_score(y,mean) if point_ok and len(y)>1 and np.var(y)>0 else np.nan,
        MAE=mean_absolute_error(y,mean) if point_ok else np.nan,
        RMSE=np.sqrt(mean_squared_error(y,mean)) if point_ok else np.nan,
        MAPE=float(np.mean(ape)) if len(ape) else np.nan,
        MdAPE=float(np.median(ape)) if len(ape) else np.nan,percentage_n=int(valid.sum()),
        PICP90=float(np.mean((y>=lo)&(y<=hi))) if interval_ok else np.nan,
        MPIW90=float(np.mean(width)) if interval_ok else np.nan,
        NMPIW90=float(np.mean(width)/np.ptp(y)) if interval_ok and np.ptp(y)>0 else np.nan,
        Winkler90=float(np.mean(winkler)) if interval_ok else np.nan,
        WIS90=float(np.mean((.5*np.abs(y-median)+.05*winkler)/1.5)) if interval_ok and median_ok else np.nan)

def aggregate_scores(frame, by):
    return frame.groupby(by,dropna=False)[METRICS].agg(['mean','std']).reset_index()
