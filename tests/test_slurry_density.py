import sys,unittest,math
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from battwin_probabilistic.slurry_density import *

class SlurryDensity(unittest.TestCase):
    def recipe(self):
        return {'Chemistry':'NMC622','Am/total weight%':57.6,'Conductive additive type':'C65',
            'Conductive additive/total weight %':1.2,'Binder1 type':'PVDF','Binder1/total weight %':1.2,
            'Binder2 (thickener)/total weight %':0.,'Dispersant additive (%)':0.,'Solvent':'NMP',
            'Solid Content (%)':60.,'Slurry density g/cm3':2.5}

    def test_96_2_2_at_60_percent(self):
        self.assertAlmostEqual(estimate_slurry_density(self.recipe())['density_g_cm3'],1/(.576/4.447+.012/1.6+.012/1.78+.4/1.03))

    def test_binders_thickener_and_dispersant_included_once(self):
        row=self.recipe();row.update({'Am/total weight%':55.8,'Binder2 (Thickener) type':'CMC',
            'Binder2 (thickener)/total weight %':.6,'Dispersent Type':'New dispersant','Dispersant additive (%)':1.2})
        r=estimate_slurry_density(row,{'New dispersant':1.4})
        self.assertAlmostEqual(sum(c['wt_pct_slurry'] for c in r['components']),100.)
        self.assertAlmostEqual(r['density_g_cm3'],1/(.558/4.447+.012/1.6+.012/1.78+.006/1.6+.012/1.4+.4/1.03))

    def test_manual_is_unchanged_and_estimate_is_explicit(self):
        original=self.recipe();self.assertEqual(apply_density(original),original)
        r=apply_density({**original,MODE:'estimated',MATERIALS:{'PVDF':1.9}})
        self.assertEqual(r[MANUAL],2.5);self.assertNotEqual(r['Slurry density g/cm3'],2.5)
        self.assertEqual(original['Slurry density g/cm3'],2.5)
        self.assertAlmostEqual(r['Slurry density g/cm3'],1/(.576/4.447+.012/1.6+.012/1.9+.4/1.03))

    def test_missing_inputs_and_invalid_densities_are_rejected(self):
        for value in [0.,-1.,math.inf,math.nan,'missing']:
            with self.subTest(value=value),self.assertRaises(ValueError):estimate_slurry_density(self.recipe(),{'NMC622':value})
        for field in ['Solvent','Chemistry','Binder2 (thickener)/total weight %']:
            row=self.recipe();del row[field]
            with self.subTest(field=field),self.assertRaises(ValueError):estimate_slurry_density(row)
        row=self.recipe();row['Solid Content (%)']=50.
        with self.assertRaisesRegex(ValueError,'solid content'):estimate_slurry_density(row)

    def test_explicit_unknown_assumption_and_case_aliases(self):
        row=self.recipe();row['Chemistry']='New active material'
        record=estimate_slurry_density(row)
        self.assertTrue(record['components'][0]['fallback']);self.assertEqual(record['components'][0]['density_g_cm3'],1.6)
        self.assertTrue(estimate_slurry_density(row,{'New active material':1.6})['components'][0]['fallback'])
        self.assertFalse(estimate_slurry_density(row,{'New active material':3.})['components'][0]['fallback'])
        record=estimate_slurry_density(self.recipe(),{'pvdf':1.9,'NMC622':4.447})
        self.assertFalse(record['components'][0]['edited'])
        self.assertTrue(record['components'][2]['edited'])
        self.assertIn('User-supplied',record['components'][2]['assumption'])
        row=self.recipe();lower={c.lower():v for c,v in row.items()}
        self.assertEqual(estimate_slurry_density(row),estimate_slurry_density(lower))

if __name__=='__main__':unittest.main()
