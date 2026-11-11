"""Regression guards for density-aware conditional process inference."""
import sys,unittest
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from battwin_probabilistic import Project
from battwin_probabilistic.promoted_inputs import MASS,DRY_T,DRY_P

@unittest.skipUnless((PROJECT/'artifacts/models/coating_drying.joblib').exists(),'Pretrained artifacts are not installed')
class ConditionalInference(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.project=Project(PROJECT)

    def test_measured_coating_targets_cannot_leak_into_particles(self):
        model=self.project.model('coating_drying')
        frame=self.project.data('coating_drying')['test'].head(2)
        expected=model.sample(frame,n_particles=40,seed=71)
        changed=frame.copy()
        changed[MASS]=9999.;changed[DRY_T]=9999.;changed[DRY_P]=-9999.
        pd.testing.assert_frame_equal(expected,model.sample(changed,n_particles=40,seed=71),check_exact=True)

    def test_upstream_dry_porosity_drives_only_porosity_target(self):
        model=self.project.model('calendaring')
        a=self.project.data('calendaring')['test'].head(6).copy();b=a.copy()
        a[DRY_P]=35.;b[DRY_P]=80.
        features=model.target_frame(a,'Calendered porosity (%)')
        np.testing.assert_allclose(features.dry_bulk_solid_fraction,1-features.dry_porosity_consistent/100)
        qa=model.quantiles(a);qb=model.quantiles(b)
        np.testing.assert_array_equal(qa['Calendered thickness (um)'],qb['Calendered thickness (um)'])
        self.assertFalse(np.allclose(qa['Calendered porosity (%)'],qb['Calendered porosity (%)']))

if __name__=='__main__':unittest.main()
