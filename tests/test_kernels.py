import unittest,sys
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from battwin_probabilistic.data import annotate,grouped_split,assert_disjoint
from battwin_probabilistic.calibration import CDFMap
from battwin_probabilistic.distributions import Mixture
from battwin_probabilistic.design_space import pareto_mask,balanced_knee,derived_shear
from battwin_probabilistic.scrap import policy_masks,assess_scrap
from battwin_probabilistic.validation import experimental_oot
from battwin_probabilistic.metrics import scores

class ScientificKernels(unittest.TestCase):
    def test_repetitions_stay_together(self):
        frame=pd.DataFrame({'x':np.repeat(np.arange(20),3),'y':np.arange(60)/10})
        schema=dict(features=['x'],targets=['y'],split_group_features=['x'])
        frame=annotate(frame,schema);a,b=grouped_split(frame,.2,42)
        self.assertEqual(len(a)+len(b),60);assert_disjoint(a,b)
        self.assertTrue(a.groupby('condition_id').size().eq(3).all())
        self.assertTrue(b.groupby('condition_id').size().eq(3).all())

    def test_leakage_is_rejected(self):
        frame=annotate(pd.DataFrame({'x':[1,1,2],'y':[2,3,4]}),dict(features=['x'],targets=['y'],split_group_features=['x']))
        with self.assertRaises(ValueError):assert_disjoint(frame.iloc[:1],frame.iloc[1:2])

    def test_pareto_ties_and_knee(self):
        np.testing.assert_array_equal(pareto_mask([[1,1],[2,1],[1,2],[.5,.5],[np.nan,9],[2,1]]),[False,True,True,False,False,True])
        index,_=balanced_knee([[1,3],[2.5,2.5],[3,1]])
        self.assertEqual(index,1)

    def test_shear_conventions(self):
        a={'Shear rate':100.,'Coating Gap (micron)':100.,'Coating Speed (m/min)':.6,'Coating ratio (%)':150.}
        self.assertAlmostEqual(derived_shear(a,1.2,100,120),200)
        a['Shear rate']=100/1.5
        self.assertAlmostEqual(derived_shear(a,1.2,100,120),200/1.2)
        a['Shear rate']=999
        with self.assertRaises(ValueError):derived_shear(a,1.2,100,120)

    def test_left_inverse_of_plateau(self):
        mapper=CDFMap([0,.2,.5,.7,1],[0,.3,.3,.8,1])
        self.assertAlmostEqual(float(mapper.inverse(.3)),.2)
        self.assertGreater(float(mapper.inverse(.31)),.5)
        levels=np.linspace(.01,.99,99)
        np.testing.assert_allclose(mapper(mapper.inverse(levels)),levels,atol=1e-12)

    def test_eight_analytic_distributions(self):
        records={'Gaussian':dict(loc=1.,scale=.5),'Gamma':dict(concentration=2.,rate=1.),
            'StudentT':dict(df=4.,loc=1.,scale=.5),'Gumbel':dict(loc=1.,scale=.5),
            'LogNormal':dict(loc=.5,scale=.3),'Weibull':dict(scale=3.,concentration=2.),
            'Laplace':dict(loc=1.,scale=.5),'Cauchy':dict(loc=1.,scale=.5)}
        for dist,params in records.items():
            with self.subTest(distribution=dist):
                mix=Mixture(dist,'raw',{k:np.full((20,2),v) for k,v in params.items()})
                p=np.broadcast_to([.05,.5,.95],(2,3))
                np.testing.assert_allclose(mix.cdf_z(mix.quantile_z(p)),p,atol=2e-7)

    def test_metrics_use_original_units(self):
        m=scores([1,2],dict(mean=np.array([1,2]),q50=np.array([1,2]),q05=np.array([.8,1.8]),q95=np.array([1.2,2.2])))
        self.assertEqual(m['R2'],1);self.assertEqual(m['MAE'],0);self.assertEqual(m['PICP90'],1)
        self.assertAlmostEqual(m['WIS90'],.05*.4/1.5)

    def test_first_rejection_and_recovery(self):
        names=['Mixing','Drying','Calendaring','Electrochemical']
        bad={s:np.zeros(4,bool) for s in names};bad['Mixing'][0]=True;bad['Calendaring'][[0,1]]=True;bad['Electrochemical'][2]=True
        first,history,direct,flag=policy_masks(bad,bad['Electrochemical'])
        self.assertEqual(first.tolist(),['Mixing','Calendaring','Electrochemical',''])
        self.assertEqual(direct.tolist(),[False,False,False,True]);self.assertEqual(flag.tolist(),[True,True,False,True])

    def test_scrap_stage_accounting(self):
        data={'path_id':0,'viscosity':1.,'Mass loading dry (mg/cm2)':10.,'Dry Coating Thickness (\u00b5m)':100.,
            'Dried Porosity (%)':70.,'Calendered thickness (um)':80.,'Calendered porosity (%)':65.,
            'Gravimetric capacity (mA h/g)':150.,'Volumetric capacity (mA h/cm3)':400.,
            'Coating Gap (micron)':120.,'Calendaring Gap (um)':60.,'Solid density reference (g/cm3)':4.447}
        p=pd.DataFrame({k:np.full(100,v) for k,v in data.items()})
        p.loc[0,'viscosity']=100.;p.loc[1,'Gravimetric capacity (mA h/g)']=300.
        result=assess_scrap(p,[150,400],[(.5,.05)])
        r=result['ranking'].iloc[0]
        self.assertAlmostEqual(r.direct_yield_pct,98.);self.assertAlmostEqual(r.flagging_yield_pct,99.)
        self.assertAlmostEqual(r.recoverable_pct_original,1.)

    def test_experimental_reference_excludes_singletons(self):
        obs=pd.DataFrame({'condition':[1,1,1,2],'Grav':[95.,100.,105.,200.],'Vol':[190.,200.,210.,400.]})
        particles=pd.DataFrame({'scenario_index':np.repeat(np.arange(4),2),'Grav':100.,'Vol':200.})
        a=experimental_oot(obs,{'chain':{'calibrated':particles}},['condition'],['condition'],['Grav','Vol'],tolerances=[5],bootstrap=30)
        self.assertEqual(a['eligibility'].eligible.sum(),1)
        self.assertEqual(a['summary'].n_observations.max(),3)
        self.assertEqual(a['summary'].observed_oot_pct.max(),0)
        b=experimental_oot(obs,{'chain':{'calibrated':particles}},['condition'],['condition'],['Grav','Vol'],tolerances=[5],reference='leave_one_out',bootstrap=30)
        self.assertAlmostEqual(b['summary'].observed_oot_pct.max(),200/3)

    def test_joint_probability_uses_pairs(self):
        obs=pd.DataFrame({'condition':[1,1,1],'Grav':[100.]*3,'Vol':[200.]*3})
        particles=pd.DataFrame({'scenario_index':np.repeat(np.arange(3),2),'Grav':[80.,100.]*3,'Vol':[200.,240.]*3})
        r=experimental_oot(obs,{'chain':{'calibrated':particles}},['condition'],['condition'],['Grav','Vol'],tolerances=[5],bootstrap=30)
        summary=r['summary'].set_index('event')
        self.assertEqual(summary.loc['Grav','predicted_oot_pct'],50.)
        self.assertEqual(summary.loc['Vol','predicted_oot_pct'],50.)
        self.assertEqual(summary.loc['Joint','predicted_oot_pct'],100.)

if __name__=='__main__':unittest.main()
