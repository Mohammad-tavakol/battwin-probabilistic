"""Publication exports; return figures without a GUI or notebook dependency."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

def axes():
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman'],'font.size':11,
        'axes.labelsize':14,'xtick.labelsize':11,'ytick.labelsize':11,'pdf.fonttype':42,'axes.unicode_minus':False})
    fig,ax=plt.subplots(figsize=(6,4.5));fig.subplots_adjust(left=.16,right=.96,bottom=.18,top=.95)
    ax.grid(ls=':',lw=.6,alpha=.35);ax.set_axisbelow(True)
    return fig,ax

def save(fig,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if path.suffix:path=path.with_suffix('')
    fig.savefig(path.with_suffix('.pdf'));fig.savefig(path.with_suffix('.png'),dpi=400)
    return path.with_suffix('.pdf')

def parity(observed,predicted,label,log=False,path=None):
    fig,ax=axes();ax.scatter(observed,predicted,s=24,alpha=.65,edgecolors='white',linewidths=.35)
    values=np.r_[observed,predicted];finite=values[np.isfinite(values)]
    if log:
        positive=finite[finite>0]
        if not len(positive):raise ValueError('No positive values for logarithmic parity')
        limits=(positive.min(),positive.max());ax.set_xscale('log');ax.set_yscale('log')
    else:limits=(finite.min(),finite.max())
    ax.plot(limits,limits,'k--',lw=1);ax.set(xlabel='Measured '+label,ylabel='Predicted '+label)
    if path:save(fig,path)
    return fig

def residuals(observed,predicted,label,log=False,path=None):
    fig,ax=axes();observed=np.asarray(observed,float);predicted=np.asarray(predicted,float)
    ax.scatter(predicted,observed-predicted,s=24,alpha=.65,edgecolors='white',linewidths=.35)
    if log:ax.set_xscale('log')
    ax.axhline(0,color='.25',ls='--',lw=1)
    ax.set(xlabel='Predicted '+label,ylabel='Residual (measured - predicted)')
    if path:save(fig,path)
    return fig

def calibration(pit,path=None):
    values=np.sort(np.asarray(pit,float));levels=np.linspace(0,1,101)
    if not len(values) or not np.isfinite(values).all():raise ValueError('Finite PIT observations required')
    fig,ax=axes();ax.plot(levels,np.searchsorted(values,levels,side='right')/len(values),color='#1679ad',lw=1.5)
    ax.plot([0,1],[0,1],'k--',lw=1);ax.set(xlabel='Nominal cumulative probability',ylabel='Observed cumulative frequency',xlim=(0,1),ylim=(0,1))
    if path:save(fig,path)
    return fig

def uncertainty_decomposition(table,path=None):
    fig,ax=axes();x=np.arange(len(table))
    for col,label in [('aleatoric_std','Aleatoric'),('epistemic_std','Epistemic'),('total_std','Total')]:
        ax.plot(x,table[col],lw=1.3,label=label)
    ax.set(xlabel='Sample number',ylabel='Predictive standard deviation');ax.legend(frameon=False,fontsize=10)
    if path:save(fig,path)
    return fig

def uncertainty(observed,median,lower,upper,label,log=False,path=None):
    fig,ax=axes();x=np.arange(len(median))
    ax.fill_between(x,lower,upper,color='#1679ad',alpha=.2,label='90% prediction interval')
    ax.plot(x,median,color='#1679ad',lw=1,label='Predictive median')
    if observed is not None:ax.scatter(x,observed,s=10,color='.15',label='Measured',zorder=4)
    if log:ax.set_yscale('log')
    ax.set(xlabel='Sample number',ylabel=label);ax.legend(frameon=False,fontsize=10)
    if path:save(fig,path)
    return fig

def pfi(summary,path=None):
    d=summary.sort_values('importance_mean').tail(15)
    fig,ax=axes();fig.subplots_adjust(left=.42)
    ax.barh(d.feature,d.importance_mean,xerr=d.importance_std,color='#1679ad',capsize=3)
    ax.axvline(0,color='.3',lw=.8);ax.set(xlabel='Decrease in R²',ylabel='Input feature')
    if path:save(fig,path)
    return fig

def pareto(frame,gravimetric,volumetric,path=None):
    fig,ax=axes();ax.scatter(frame[gravimetric],frame[volumetric],s=24,color='#1679ad')
    ax.set(xlabel='Gravimetric capacity (mAh/g)',ylabel='Volumetric capacity (mAh/cm³)')
    if path:save(fig,path)
    return fig

def oot(summary,path=None):
    fig,ax=axes()
    observed=summary.drop_duplicates('tolerance_pct').sort_values('tolerance_pct')
    x=observed.tolerance_pct.to_numpy(float)
    ax.fill_between(x,observed.observed_low_pct.to_numpy(float),observed.observed_high_pct.to_numpy(float),color='.65',alpha=.2)
    ax.plot(x,observed.observed_oot_pct,'ko-',label='Observed')
    for method,d in summary.groupby('method'):
        d=d.sort_values('tolerance_pct');ax.plot(d.tolerance_pct,d.predicted_oot_pct,'D-',label=method)
    ax.set(xlabel='Tolerance (%)',ylabel='Condition-balanced OOT (%)',ylim=(-2,103),xticks=x)
    ax.legend(frameon=False,fontsize=10)
    if path:save(fig,path)
    return fig
