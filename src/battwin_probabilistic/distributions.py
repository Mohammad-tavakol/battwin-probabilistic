"""Analytic equally weighted XGBoostLSS mixtures on the model target scale."""
import numpy as np
from scipy import special

class Mixture:
    """Equally weighted analytic member CDFs on the model target scale.

The inverse target transformation clips at zero. The CDF therefore
retains any probability mass at zero; positive observations use the analytic CDF.
"""
    def __init__(self,distribution,mode,parameters):
        self.distribution=distribution;self.mode=mode
        self.parameters={k:np.asarray(v,float) for k,v in parameters.items()}
        self.n=next(iter(self.parameters.values())).shape[1]

    @classmethod
    def predict(cls,ensemble,frame):
        import xgboost as xgb
        values=[]
        for member in ensemble['members']:
            member['model'].booster.set_param({'nthread':1})
            transformed=member['preprocessor'].transform(frame)
            if hasattr(transformed,'toarray'):transformed=transformed.toarray()
            values.append(member['model'].predict(xgb.DMatrix(transformed),pred_type='parameters'))
        columns=list(values[0].columns)
        return cls(ensemble['distribution'],ensemble['target_mode'],{c:np.stack([v[c].to_numpy(float) for v in values]) for c in columns})

    def subset(self,indices):
        return Mixture(self.distribution,self.mode,{k:v[:,indices] for k,v in self.parameters.items()})

    def record(self):return {'distribution':self.distribution,'mode':self.mode,'parameters':self.parameters}

    @classmethod
    def restore(cls,d):return cls(d['distribution'],d['mode'],d['parameters'])

    def to_original(self,z):
        if self.mode=='log1p':
            with np.errstate(over='ignore'):z=np.expm1(z)
        assert np.isfinite(z).all(),'Nonfinite inverse target transform; no tail silently discarded.'
        return np.maximum(0,z)

    def _arguments(self,k=1):return {c:v[:,:,None] for c,v in self.parameters.items()}

    def _cdf_components(self,z):
        a=self._arguments();z=np.asarray(z,float)[None,:,:]
        if self.distribution=='Gamma':return special.gammainc(a['concentration'],np.maximum(0,z)*a['rate'])
        if self.distribution=='Gaussian':return special.ndtr((z-a['loc'])/a['scale'])
        if self.distribution=='StudentT':return special.stdtr(a['df'],(z-a['loc'])/a['scale'])
        if self.distribution=='Gumbel':return np.exp(-np.exp(np.clip(-(z-a['loc'])/a['scale'],-700,700)))
        if self.distribution=='LogNormal':return np.where(z<=0,0,special.ndtr((np.log(np.maximum(z,np.nextafter(0.,1.)))-a['loc'])/a['scale']))
        if self.distribution=='Weibull':return -np.expm1(-np.maximum(z/a['scale'],0)**a['concentration'])
        if self.distribution=='Laplace':
            t=(z-a['loc'])/a['scale'];return np.where(t<0,.5*np.exp(np.minimum(t,0)),1-.5*np.exp(-np.maximum(t,0)))
        if self.distribution=='Cauchy':return .5+np.arctan((z-a['loc'])/a['scale'])/np.pi
        raise ValueError(self.distribution)

    def cdf_z(self,z):
        z=np.asarray(z,float);one=z.ndim==1
        if one:z=z[:,None]
        result=self._cdf_components(z).mean(axis=0)
        return result[:,0] if one else result

    def cdf(self,y):
        y=np.asarray(y,float)
        z=np.log1p(np.maximum(y,0)) if self.mode=='log1p' else y
        return np.where(y<0,0,self.cdf_z(z))

    def _ppf_components(self,p):
        a=self._arguments();p=np.asarray(p,float)[None,:,:]
        if self.distribution=='Gamma':return special.gammaincinv(a['concentration'],p)/a['rate']
        if self.distribution=='Gaussian':return a['loc']+a['scale']*special.ndtri(p)
        if self.distribution=='StudentT':return a['loc']+a['scale']*special.stdtrit(a['df'],p)
        if self.distribution=='Gumbel':return a['loc']-a['scale']*np.log(-np.log(p))
        if self.distribution=='LogNormal':return np.exp(a['loc']+a['scale']*special.ndtri(p))
        if self.distribution=='Weibull':return a['scale']*(-np.log1p(-p))**(1/a['concentration'])
        if self.distribution=='Laplace':return a['loc']-a['scale']*np.sign(p-.5)*np.log1p(-2*np.abs(p-.5))
        if self.distribution=='Cauchy':return a['loc']+a['scale']*np.tan(np.pi*(p-.5))
        raise ValueError(self.distribution)

    def quantile_z(self,p):
        p=np.asarray(p,float)
        if p.ndim==0:p=np.full((self.n,1),p)
        elif p.ndim==1:p=p[:,None]
        p=np.clip(p,1e-12,1-1e-12)
        brackets=self._ppf_components(p)
        lo=brackets.min(axis=0);hi=brackets.max(axis=0)
        logarithmic=self.distribution=='Gamma'
        if logarithmic:
            # Small Gamma shapes can place quantiles far below 1e-12. Linear
            # bisection would lose that part of the CDF even in ordinary units.
            lo=np.log(np.maximum(lo,np.nextafter(0.,1.)))
            hi=np.log(np.maximum(hi,np.nextafter(0.,1.)))
        for _ in range(45):
            mid=(lo+hi)/2;value=np.exp(mid) if logarithmic else mid
            below=self.cdf_z(value)<p
            lo=np.where(below,mid,lo);hi=np.where(below,hi,mid)
        result=(lo+hi)/2
        return np.exp(result) if logarithmic else result

    def quantile(self,p):return self.to_original(self.quantile_z(p))

    def sample_component_latent(self,u,member_ids):
        u=np.asarray(u,float)[:,None]
        q=self._ppf_components(np.clip(u,1e-12,1-1e-12))[:,:,0]
        return q[np.asarray(member_ids,int),np.arange(self.n)]

def probability_cdf(mixture,y,method,scale=1.,mapper=None,median=None):
    if method=='Kuleshov':return mapper(mixture.cdf(y))
    if method=='Uncalibrated':return mixture.cdf(y)
    if median is None:median=mixture.quantile(.5)[:,0]
    y=np.asarray(y,float)
    return np.where(y<0,0,mixture.cdf(median+(y-median)/scale))

def prediction_quantiles(mixture,levels,method,scale=1.,mapper=None,median=None):
    levels=np.asarray(levels,float)
    p=np.broadcast_to(levels,(mixture.n,len(levels)))
    if method=='Kuleshov':p=mapper.inverse(p)
    q=mixture.quantile(p)
    if method not in ['Uncalibrated','Kuleshov']:
        if median is None:median=mixture.quantile(.5)[:,0]
        q=np.maximum(0,median[:,None]+scale*(q-median[:,None]))
    return q
