"""Held-out monotone CDF calibration, preserving the existing map convention."""
import numpy as np
from sklearn.isotonic import IsotonicRegression

class CDFMap:
    def __init__(self,x,y):
        self.x=np.asarray(x,float);self.y=np.asarray(y,float)
        assert np.all(np.diff(self.x)>0) and np.all(np.diff(self.y)>=0)
        assert tuple(self.x[[0,-1]])==(0.,1.) and tuple(self.y[[0,-1]])==(0.,1.)

    @classmethod
    def fit(cls,pit):
        pit=np.asarray(pit,float)
        assert np.isfinite(pit).all() and np.all((pit>=0)&(pit<=1))
        # Upper empirical ranks, including ties, as in Algorithm 1.
        ranks=np.searchsorted(np.sort(pit),pit,side='right')/len(pit)
        numerical_pit=np.clip(pit,1e-12,1-1e-12)
        iso=IsotonicRegression(y_min=0,y_max=1,increasing=True,out_of_bounds='clip')
        iso.fit(numerical_pit,ranks)
        return cls(np.r_[0,iso.X_thresholds_,1],np.r_[0,iso.y_thresholds_,1])

    def __call__(self,p):return np.interp(p,self.x,self.y,left=0,right=1)

    def inverse(self,p):
        # Left search selects the first point of any plateau when p equals its
        # height; for p above a plateau, its last point starts the rising segment.
        # This also resolves steep endpoint segments without bisection error.
        p=np.asarray(p,float);clipped=np.clip(p,0,1)
        right=np.clip(np.searchsorted(self.y,clipped,side='left'),1,len(self.y)-1)
        left=right-1
        rise=self.y[right]-self.y[left]
        fraction=np.divide(clipped-self.y[left],rise,out=np.zeros_like(clipped),where=rise>0)
        value=self.x[left]+fraction*(self.x[right]-self.x[left])
        return np.where(p<=0,0,value)

    def record(self):return {'x':self.x.tolist(),'y':self.y.tolist(),'algorithm':'Kuleshov 2018 Algorithm 1, upper empirical PIT ranks, isotonic interpolation, endpoint anchors'}
