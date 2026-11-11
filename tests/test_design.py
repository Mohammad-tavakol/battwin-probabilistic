import unittest,sys,tempfile
from pathlib import Path
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from battwin_probabilistic import Project
from battwin_probabilistic.design_space import SearchConfig,GRAV,VOL

@unittest.skipUnless((PROJECT/'artifacts/models/mixing.joblib').exists(),'Local pretrained artifacts are not installed')
class DesignWorkflow(unittest.TestCase):
    def test_search_sampling_verification_and_cache(self):
        from battwin_probabilistic.optimisation import Experiment
        project=Project(PROJECT)
        with tempfile.TemporaryDirectory() as folder:
            config=SearchConfig(discharge_rate=.2,screen_particles=40,refine_particles=40,verify_particles_per_seed=40)
            experiment=Experiment(project.data('electrochemical')['development'],config,project.catalog,folder)
            observed=experiment.observed_paths().head(2)
            generated=experiment.generate(2)
            self.assertTrue((generated['Calendaring Gap (um)']<generated['Coating Gap (micron)']).all())
            summary=experiment.evaluate(observed,40,42,'smoke',.1)
            cached=experiment.evaluate(observed,40,42,'smoke',.1)
            pd.testing.assert_frame_equal(summary,cached)
            experiment.target=summary.iloc[0][[GRAV+'__mean',VOL+'__mean']].to_numpy(float)
            checked,particles=experiment.verify(observed.head(1),'smoke_verify',[101,201])
            self.assertEqual(len(particles),80)
            self.assertEqual(particles.verification_seed.nunique(),2)
            expected=checked.eligible&checked.mean_matches_target&checked.both_seed_means_match&checked.both_seed_eligibility
            self.assertTrue(checked.accepted.equals(expected))

    def test_test_data_cannot_define_design_space(self):
        from battwin_probabilistic.optimisation import Experiment
        project=Project(PROJECT)
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                Experiment(project.data('electrochemical')['test'],catalog=project.catalog,output_dir=folder)

    def test_simulation_only_sensitivity(self):
        from battwin_probabilistic.sensitivity import compute_chain_pfi,summarise_chain_pfi
        project=Project(PROJECT);chain=project.chain()
        frame=project.data('electrochemical')['train'].iloc[:3]
        raw=compute_chain_pfi(chain,frame,features=['Coating Gap (micron)'],n_repeats=1,
            n_particles_override=40,n_member_repeats=2,pfi_output_columns='final_kpis',max_scenarios=None)
        self.assertFalse(summarise_chain_pfi(raw).empty)

    def test_calculated_porosity_and_variance(self):
        from battwin_probabilistic.physics import porosity_from_thickness
        from battwin_probabilistic.uncertainty import decompose
        np.testing.assert_allclose(porosity_from_thickness([10.],[50.],[4.]),[50.])
        project=Project(PROJECT);m=project.model('mixing')
        values=decompose(m,project.data('mixing')['test'].head(1),particles=40)
        np.testing.assert_allclose(values.total_std**2,values.aleatoric_std**2+values.epistemic_std**2)

if __name__=='__main__':unittest.main()
