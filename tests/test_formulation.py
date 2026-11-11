import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
PROJECT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(PROJECT/'src'))
from battwin_probabilistic import Project,SimulationConfig,COMPONENTS,SOLIDS,calculated_solids,prepare_formulation
from battwin_probabilistic.design_space import SearchConfig
from battwin_probabilistic.models import ProcessModel

class FormulationAccounting(unittest.TestCase):
    def test_sum_aliases_and_stale_or_omitted_solid_content(self):
        row=dict(zip(COMPONENTS,[60.,2.,1.,.5]));row['Dispersant additive (%)']=.75
        frame=pd.DataFrame([row],index=[8]);result=prepare_formulation(frame)
        self.assertNotIn(SOLIDS,frame);self.assertEqual(result.loc[8,SOLIDS],63.5)
        aliased=frame.rename(columns={c:c.lower() for c in COMPONENTS}).assign(**{SOLIDS:99.})
        self.assertEqual(prepare_formulation(aliased).iloc[0][SOLIDS],63.5)
        self.assertEqual(calculated_solids(result.iloc[0]),63.5)
        np.testing.assert_array_equal(result[list(COMPONENTS)].iloc[0],[60,2,1,.5])

    def test_invalid_and_partial_formulations(self):
        for amounts in [[0,0,0,0],[98,2,1,0],[float('nan'),1,1,0],[-1,1,1,1],[float('inf'),1,1,0]]:
            with self.subTest(amounts=amounts),self.assertRaises(ValueError):prepare_formulation(pd.DataFrame([dict(zip(COMPONENTS,amounts))]))
        partial=pd.DataFrame({SOLIDS:[70.]})
        pd.testing.assert_frame_equal(prepare_formulation(partial,require_complete=False),partial)
        with self.assertRaisesRegex(ValueError,'all formulation'):prepare_formulation(partial)
        row={**dict(zip(COMPONENTS,[60,1,1,1])),'Binder2 (Thickener) type':'none'}
        with self.assertRaisesRegex(ValueError,'Not used'):prepare_formulation(pd.DataFrame([row]))

    def test_individual_model_derives_from_auxiliary_components(self):
        model=ProcessModel.__new__(ProcessModel);model.features=[SOLIDS,'x'];model.stage='example';model.bundle={}
        frame=pd.DataFrame([{**dict(zip(COMPONENTS,[60.,2.,1.,.5])),SOLIDS:10.,'x':3.}])
        result=model._frame(frame)
        self.assertEqual(result.columns.tolist(),[SOLIDS,'x']);self.assertEqual(result.iloc[0][SOLIDS],63.5)
        self.assertEqual(frame.iloc[0][SOLIDS],10.)

    def test_process_pfi_uses_components_when_solid_content_is_derived(self):
        from battwin_probabilistic.pfi import process_importance
        model=ProcessModel.__new__(ProcessModel);model.features=[SOLIDS];model.targets=['y']
        frame=pd.DataFrame([{**dict(zip(COMPONENTS,[60,2,1,0])),SOLIDS:63.,'y':1.}])
        with patch('battwin_probabilistic.pfi.permutation_importance',return_value=('raw','summary')) as calculate:
            process_importance(model,frame)
            self.assertEqual(calculate.call_args.args[3],list(COMPONENTS))
        from battwin_probabilistic.sensitivity import compute_chain_pfi
        with self.assertRaisesRegex(ValueError,'derived'):compute_chain_pfi(None,frame,features=[SOLIDS])

@unittest.skipUnless((PROJECT/'artifacts/models/mixing.joblib').is_file(),'Pretrained artifacts unavailable')
class FormulationWorkflows(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.project=Project(PROJECT);cls.development=cls.project.data('electrochemical')['development']

    def experiment(self,folder,**kwargs):
        from battwin_probabilistic.optimisation import Experiment
        return Experiment(self.development,SearchConfig(discharge_rate=.2,screen_particles=20,refine_particles=20,verify_particles_per_seed=20,**kwargs),self.project.catalog,folder)

    def test_chain_and_process_use_corrected_solids_without_changing_inputs(self):
        frame=self.development.head(1).copy();frame[SOLIDS]=1.
        chain=self.project.chain(SimulationConfig(particles=20));corrected=prepare_formulation(frame)
        _,actual=chain.simulate(frame);_,expected=chain.simulate(corrected.drop(columns=SOLIDS))
        pd.testing.assert_frame_equal(actual,expected[actual.columns])
        self.assertEqual(frame.iloc[0][SOLIDS],1.)
        model=self.project.model('electrochemical')
        pd.testing.assert_frame_equal(model.predict(frame),model.predict(corrected))

    def test_optimisation_and_inverse_sweep_fractions_and_recalculate_solids(self):
        from battwin_probabilistic.inverse_design import generate_multiscale
        with tempfile.TemporaryDirectory() as folder:
            exp=self.experiment(folder,formulation_bounds={COMPONENTS[0]:(64.,64.2),COMPONENTS[1]:(1.331,1.339)})
            initial=exp.generate(5);inverse=generate_multiscale(exp,initial,4,1)
            for frame in [initial,inverse]:
                self.assertGreater(frame[COMPONENTS[0]].nunique(),1)
                self.assertTrue(frame[COMPONENTS[0]].between(64.,64.2).all())
                self.assertTrue(frame[COMPONENTS[1]].between(1.331,1.339).all())
                np.testing.assert_allclose(frame[SOLIDS],frame[list(COMPONENTS)].sum(axis=1),atol=1e-12)
            stale=initial.head(1).copy();stale[SOLIDS]=1.
            summary=exp.evaluate(stale,20,42,'formulation_test',1.)
            self.assertAlmostEqual(summary.iloc[0][SOLIDS],calculated_solids(stale.iloc[0]))
            exp.target=summary.iloc[0][['Gravimetric capacity (mA h/g)__mean','Volumetric capacity (mA h/cm3)__mean']].to_numpy(float)
            verified,particles=exp.verify(initial.head(1),'verify_formulation',[101,201])
            self.assertEqual(len(particles),40);self.assertEqual(verified.iloc[0][SOLIDS],calculated_solids(verified.iloc[0]))
            domain=pd.read_csv(Path(folder)/'reports/design_space.csv').set_index('parameter')
            self.assertEqual(domain.loc[SOLIDS,'mode'],'derived')
            self.assertEqual(domain.loc[COMPONENTS[0],'mode'],'continuous')
            self.assertEqual(domain.loc[COMPONENTS[0],'minimum'],64.)
            self.assertEqual(domain.loc[COMPONENTS[0],'maximum'],64.2)

    def test_independent_solid_content_and_unsupported_ranges_are_rejected(self):
        cases=[dict(formulation_bounds={SOLIDS:(60,70)}),dict(formulation_bounds={COMPONENTS[0]:(90,99)}),
            dict(formulation_bounds={COMPONENTS[0]:(-1,3)}),dict(vary_formulation=False,formulation_bounds={COMPONENTS[0]:(64,64.2)})]
        for config in cases:
            with self.subTest(config=config),tempfile.TemporaryDirectory() as folder,self.assertRaises(ValueError):self.experiment(folder,**config)

    def test_legacy_inverse_results_cannot_skip_new_calculation(self):
        with tempfile.TemporaryDirectory() as folder:
            exp=self.experiment(folder)
            (Path(folder)/'inverse_complete.json').write_text(json.dumps(dict(accepted=3)))
            with self.assertRaisesRegex(ValueError,'fresh output directory'):exp.inverse()

if __name__=='__main__':unittest.main()
