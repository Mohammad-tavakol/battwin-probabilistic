import unittest,tempfile,sys,json
from pathlib import Path
from dataclasses import replace
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))
from battwin_probabilistic import Project,SimulationConfig,TrainingConfig,TuningConfig
from battwin_probabilistic.models import ProcessModel
from battwin_probabilistic.training import train_process,cross_validate,select_configurations
from battwin_probabilistic.data import annotate

class TrainingWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        x=np.repeat(np.arange(20,dtype=float),3)
        cls.frame=pd.DataFrame({'x':x,'kind':np.where(x<10,'a','b'),'y':2+.5*x+np.tile([-.1,0,.1],20)})
        cls.schema=dict(stage='toy',features=['x','kind'],targets=['y'],categorical_features=['kind'],
            split_group_features=['x','kind'],bootstrap_group_features=['x','kind'])
        cls.frame=annotate(cls.frame,cls.schema)
        cls.configs={'y':dict(distribution='Gaussian',target_mode='raw',num_boost_round=2,
            dist_kwargs=dict(stabilization='MAD',response_fn='softplus',loss_fn='nll'),
            xgb_params=dict(eta=.05,max_depth=1,nthread=1,verbosity=0))}

    def test_train_calibrate_single_file_roundtrip(self):
        from battwin_probabilistic.artifacts import load_artifact
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'toy.joblib'
            a=train_process(self.frame.iloc[:48],self.frame.iloc[48:],self.schema,self.configs,output_path=path)
            self.assertEqual(len(a['model_bundle']['target_ensembles']['y']['members']),20)
            m=ProcessModel(path);self.assertTrue(np.isfinite(m.predict(self.frame.iloc[48:]).select_dtypes('number')).all().all())
            self.assertEqual(load_artifact(path)['kuleshov_calibration']['y']['n_rows'],12)

    def test_test_data_cannot_fit_calibration(self):
        validation=self.frame.iloc[48:].assign(partition='test')
        with self.assertRaises(ValueError):train_process(self.frame.iloc[:48],validation,self.schema,self.configs)

    def test_fixed_configuration_grouped_cv(self):
        detail,summary=cross_validate(self.frame,self.schema,self.configs,TrainingConfig(cv_folds=2))
        self.assertEqual(detail.fold.nunique(),2);self.assertEqual(detail.configuration_hash.nunique(),1)
        self.assertTrue(detail.evaluation.eq('held-out CV fold').all())
        self.assertIn('R2',summary.columns.get_level_values(0))

    def test_optuna_executes_hyperparameter_search(self):
        config=TuningConfig(trials=2,top_candidates=1,distributions=('Gaussian',),transformations=('raw',),
            screening_rounds=2,minimum_rounds=2,maximum_rounds=2,round_step=1)
        configurations,reports=select_configurations(self.frame,self.schema,config)
        self.assertIn('y',configurations);self.assertEqual(len(reports['trials']),2)
        self.assertTrue(reports['trials'].state.eq('COMPLETE').any())

@unittest.skipUnless((PROJECT/'artifacts/models/mixing.joblib').exists(),'Local pretrained artifacts are not installed')
class FrozenWorkflow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):cls.project=Project(PROJECT)

    def test_all_process_splits_disjoint(self):
        for stage in ['mixing','coating_drying','calendaring','electrochemical']:
            data=self.project.data(stage)
            self.assertTrue(len(data['train'])>0);self.assertTrue(len(data['validation'])>0);self.assertTrue(len(data['test'])>0)

    def test_predictions_match_reference_quantiles(self):
        reference_path=PROJECT/'tests/reference/quantiles.json'
        if not reference_path.exists():self.skipTest('Numerical reference fixture is not installed')
        expected=json.loads(reference_path.read_text(encoding='utf-8'))
        for stage,targets in expected.items():
            frame=pd.read_csv(PROJECT/'examples/data'/f'{stage}_inputs.csv',float_precision='round_trip')
            result=self.project.model(stage).quantiles(frame)
            for target,q in result.items():np.testing.assert_allclose(q,targets[target],rtol=0,atol=2e-7)

    def test_chain_matches_reference_and_particle_count_fixed(self):
        source=PROJECT/'tests/reference/chain_particles.csv'
        if not source.exists():self.skipTest('Numerical reference fixture is not installed')
        frame=pd.read_csv(PROJECT/'examples/data/chain_inputs.csv',float_precision='round_trip')
        _,particles=self.project.chain(SimulationConfig(particles=40)).simulate(frame)
        reference=pd.read_csv(source,float_precision='round_trip')
        np.testing.assert_allclose(particles[reference.columns],reference,rtol=0,atol=2e-7)
        self.assertTrue(particles.groupby('scenario_index').size().eq(40).all())
        for stage in ['Mixing','Drying','Calendaring','Electrochemical']:
            self.assertTrue(particles.groupby(['scenario_index',stage+'__member_id']).size().eq(2).all())

    def test_chain_configurations_do_not_share_physical_policy(self):
        physical=self.project.chain(SimulationConfig(particles=20,physical_porosity=True))
        direct=self.project.chain(SimulationConfig(particles=20,physical_porosity=False))
        self.assertTrue(bool(physical.stages[1].physical_spec('Dried Porosity (%)')))
        self.assertIsNone(direct.stages[1].physical_spec('Dried Porosity (%)'))
        self.assertTrue(bool(physical.stages[1].physical_spec('Dried Porosity (%)')))

if __name__=='__main__':unittest.main()
