"""Safeguarded analytic mixture inverse, with explicit CDF residual checks.

Same distribution and endpoint conventions as calibration.Mixture. Newton steps
only accelerate root finding; brackets and bisection handle slow/extreme roots.
"""
import numpy as np
from scipy import special
from .distributions import Mixture

REFERENCE_INVERSE=Mixture.quantile_z
REFERENCE_SAMPLE=Mixture.sample_component_latent

def component_sample(self,u,member_ids):
    ids=np.asarray(member_ids,int);rows=np.arange(self.n)
    one=Mixture(self.distribution,self.mode,{k:v[ids,rows][None,:] for k,v in self.parameters.items()})
    return one._ppf_components(np.clip(np.asarray(u,float)[:,None],1e-12,1-1e-12))[0,:,0]

def density(self,z,logarithmic=False):
    a=self._arguments();v=np.asarray(z)[None,:,:]
    if self.distribution=='Gaussian':
        t=(v-a['loc'])/a['scale'];logpdf=-.5*t*t-np.log(a['scale'])-.5*np.log(2*np.pi)
    elif self.distribution=='StudentT':
        t=(v-a['loc'])/a['scale'];df=a['df']
        logpdf=special.gammaln((df+1)/2)-special.gammaln(df/2)-.5*np.log(df*np.pi)-np.log(a['scale'])-(df+1)/2*np.log1p(t*t/df)
    elif self.distribution=='Gamma':
        shape=a['concentration'];rate=a['rate'];positive=np.maximum(v,np.nextafter(0.,1.))
        logpdf=shape*np.log(rate)-special.gammaln(shape)+(shape-1)*np.log(positive)-rate*positive
    elif self.distribution=='Gumbel':
        t=(v-a['loc'])/a['scale'];logpdf=-t-np.exp(np.clip(-t,-700,700))-np.log(a['scale'])
    else:raise ValueError(self.distribution)
    if logarithmic:logpdf+=np.log(np.maximum(v,np.nextafter(0.,1.)))
    return np.exp(np.clip(logpdf,-745,700)).mean(axis=0)

def inverse(self,p):
    if self.distribution not in ['StudentT','Gamma'] or (self.distribution=='Gamma' and self.parameters['concentration'].min()<.1):
        return REFERENCE_INVERSE(self,p)
    p=np.asarray(p,float)
    if p.ndim==0:p=np.full((self.n,1),p)
    elif p.ndim==1:p=p[:,None]
    p=np.clip(p,1e-12,1-1e-12)
    if self.distribution=='Gamma':
        # Cantelli bounds bracket every component quantile without solving 20
        # component inverse-Gamma problems per particle.
        a=self._arguments();mean=a['concentration']/a['rate'];sd=np.sqrt(a['concentration'])/a['rate']
        lo=np.maximum(0,(mean-sd*np.sqrt((1-p)/p)).min(axis=0))
        hi=(mean+sd*np.sqrt(p/(1-p))).max(axis=0)
    else:
        # Standardised Student-t quantiles have ordered absolute tails in df.
        # Envelope location/scale extrema give a conservative mixture bracket.
        df=self.parameters['df'];loc=self.parameters['loc'];scale=self.parameters['scale']
        heavy=special.stdtrit(df.min(axis=0)[:,None],p)
        light=special.stdtrit(df.max(axis=0)[:,None],p)
        positive=p>=.5
        lo=loc.min(axis=0)[:,None]+np.where(positive,scale.min(axis=0)[:,None]*light,scale.max(axis=0)[:,None]*heavy)
        hi=loc.max(axis=0)[:,None]+np.where(positive,scale.max(axis=0)[:,None]*heavy,scale.min(axis=0)[:,None]*light)
    logarithmic=self.distribution=='Gamma'
    if logarithmic:
        lo=np.log(np.maximum(lo,np.nextafter(0.,1.)));hi=np.log(np.maximum(hi,np.nextafter(0.,1.)))
    shape=p.shape;levels=shape[1];pflat=p.ravel();lo=lo.ravel();hi=hi.ravel()
    value=(lo+hi)/2;active=np.ones(len(pflat),bool)
    for iteration in range(70):
        ids=np.flatnonzero(active)
        if not len(ids):break
        mix=self.subset(ids//levels)
        current=value[ids];z=np.exp(current) if logarithmic else current
        cdf=mix.cdf_z(z);error=cdf-pflat[ids]
        tolerance=np.minimum(2e-12,1e-8*np.minimum(pflat[ids],1-pflat[ids]))
        converged=(np.abs(error)<tolerance)|((hi[ids]-lo[ids])<1e-11*(1+np.abs(current)))
        active[ids[converged]]=False
        below=error<0
        lo[ids[below&~converged]]=current[below&~converged]
        hi[ids[~below&~converged]]=current[~below&~converged]
        if iteration<18:
            derivative=density(mix,z[:,None],logarithmic)[:,0]
            with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
                step=np.divide(error,derivative,out=np.full_like(error,np.inf),where=derivative>0)
            proposal=current-step
            safe=np.isfinite(proposal)&(proposal>lo[ids])&(proposal<hi[ids])
            candidate=np.where(safe,proposal,(lo[ids]+hi[ids])/2)
        else:candidate=(lo[ids]+hi[ids])/2
        value[ids[~converged]]=candidate[~converged]
    result=(np.exp(value) if logarithmic else value).reshape(shape)
    residual=np.abs(self.cdf_z(result)-p)
    # Gamma floating-point underflow can leave a quantile below the smallest
    # representable number. The reference inverse retains the same convention.
    underflow=logarithmic & (result<=np.nextafter(0.,1.)*2)
    assert np.all((residual<2e-8)|underflow),f'Mixture inverse residual {residual.max()}'
    return result

def enable():
    Mixture.quantile_z=promoted_inverse
    Mixture.sample_component_latent=component_sample


def promoted_inverse(self,p):
    """Same inverse problem, including audited flat-CDF handling for new targets."""
    if self.distribution not in ['Gaussian','Gumbel']:return inverse(self,p)
    p=np.asarray(p,float)
    if p.ndim==0:p=np.full((self.n,1),p)
    elif p.ndim==1:p=p[:,None]
    p=np.clip(p,1e-12,1-1e-12)
    if p.shape[1]>1:
        nlevels=p.shape[1]
        return promoted_inverse(self.subset(np.repeat(np.arange(self.n),nlevels)),p.ravel()).reshape(p.shape)
    q=self._ppf_components(p);low=q.min(axis=0)[:,0];high=q.max(axis=0)[:,0];target=p[:,0]
    value=q.mean(axis=0)[:,0];active=np.ones(self.n,dtype=bool)
    for iteration in range(70):
        ids=np.flatnonzero(active)
        if not len(ids):break
        sub=self.subset(ids);v=value[ids];error=sub.cdf_z(v)-target[ids]
        derivative=density(sub,v[:,None],False)[:,0]
        location_tolerance=1e-11*(1+np.abs(v))
        with np.errstate(divide='ignore',invalid='ignore'):estimated_error=np.abs(error)/derivative
        resolved=(np.abs(error)<2e-12)&(derivative>1e-4)&(estimated_error<location_tolerance)
        done=resolved|((high[ids]-low[ids])<location_tolerance)
        active[ids[done]]=False;below=error<0
        low[ids[below&~done]]=v[below&~done];high[ids[~below&~done]]=v[~below&~done]
        if iteration<18:
            with np.errstate(divide='ignore',invalid='ignore',over='ignore'):proposal=v-error/derivative
            safe=np.isfinite(proposal)&(proposal>low[ids])&(proposal<high[ids]);next_value=np.where(safe,proposal,(low[ids]+high[ids])/2)
        else:next_value=(low[ids]+high[ids])/2
        value[ids[~done]]=next_value[~done]
    residual=np.abs(self.cdf_z(value)-target)
    if (residual>2e-8).any():
        bad=np.flatnonzero(residual>2e-8);value[bad]=REFERENCE_INVERSE(self.subset(bad),p[bad])[:,0]
    assert np.max(np.abs(self.cdf_z(value)-target))<3e-8
    return value[:,None]
