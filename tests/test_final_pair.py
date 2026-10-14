import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from pipelines.models.final_pair import FEATURES, load_pair, score, verify_identity
from pipelines.inference.clinical_scores import scores
from pipelines.inference.final_pair import run, load_ukb

ROOT = Path(__file__).resolve().parents[1]


class FinalPairTests(unittest.TestCase):
    def setUp(self):
        self.pair, _ = load_pair(ROOT/'release')

    def fixture(self, root):
        rng = np.random.RandomState(12)
        data = pd.DataFrame(rng.normal(size=(30,len(FEATURES))),columns=FEATURES)
        values = {'age_feat':60., 'sex':1., 'SBP':130., 'DBP':80., 'TC':5., 'TG':1.5, 'HDLC':1.3, 'LDLC':3., 'BMI':25., 'height':170., 'weight':70., 'waist':85., 'smoking':0.}
        for key,value in values.items():data[key]=value+rng.normal(0,.1,30)
        data['sex']=np.arange(30)%2
        data['smoking']=np.arange(30)%2
        data['eid']=[str(i) for i in range(30)]
        enriched=data.copy()
        enriched['TC']=8.
        enriched['diabetes']=0.
        enriched['treated_hypertension']=0.
        enriched['family_history']=0.
        enriched['FBG']=5.
        enriched['eGFR']=90.
        outcomes=pd.DataFrame({'eid':data.eid,'endpoint':'ASCVD','event':np.arange(30)%2,'time':np.arange(30)/3+1})
        data.to_csv(root/'original.csv',index=False)
        enriched.to_csv(root/'enriched.csv',index=False)
        outcomes.to_csv(root/'outcomes.csv',index=False)
        much=enriched.copy()
        much['PATIENT_SN']=much.eid
        much['ASCVD_EVENT']=outcomes.event
        much['FOLLOWUP_YEARS']=outcomes.time
        much.to_csv(root/'much.csv',index=False)
        cfg={'models':str(ROOT/'release'),'private_output':'outputs','inference':{
            'ukb':{'model_features':'original.csv','score_features':'enriched.csv','outcomes':'outcomes.csv','age_bands':[[40,74]],'bootstrap':2,'jobs':1,'expected_counts':{'40-74':{'n':30,'events':15}}},
            'much':{'table':'much.csv','bootstrap':2,'jobs':1}}}
        config=root/'config.json';config.write_text(json.dumps(cfg))
        return cfg,config

    def test_frozen_identity_rejects_wave3_substitution(self):
        artifact=copy.deepcopy(self.pair['risk'])
        artifact['source_training_wave']=3.
        with self.assertRaises(ValueError):verify_identity(artifact,'risk')
        with self.assertRaises(ValueError):verify_identity(self.pair['risk'],'event')

    def test_original_and_enriched_feature_namespaces_stay_separate(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg,_=self.fixture(root)
            data=load_ukb(cfg['inference']['ukb'],lambda p:root/p)
            self.assertTrue(data['score__TC'].eq(8).all())
            self.assertFalse(data['TC'].eq(8).any())
            original=pd.read_csv(root/'original.csv')
            np.testing.assert_allclose(score(self.pair['risk'],data),score(self.pair['risk'],original))

    def test_both_cohorts_write_aggregate_files_without_predictions(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);_,config=self.fixture(root)
            with patch('pipelines.inference.final_pair.plot'):
                for cohort in ['UKB','MUCH']:
                    frame=run(config,cohort)
                    self.assertEqual(len(frame),15)
                    self.assertTrue(frame.n.eq(30).all())
                    self.assertFalse(set(frame.columns)&{'eid','time','event','score','risk'})
                    files={p.name for p in (root/'outputs'/cohort.lower()).iterdir()}
                    self.assertEqual(files,{'aggregate_metrics.csv','aggregate_summary.csv','run_manifest.json'})

    def test_count_mismatch_stops_the_analysis(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);cfg,config=self.fixture(root)
            cfg['inference']['ukb']['expected_counts']['40-74']['n']=31
            config.write_text(json.dumps(cfg))
            with self.assertRaises(ValueError):run(config,'UKB')

    def test_comparator_formulas_match_historical_reference_values(self):
        frame=pd.DataFrame({'age_feat':[40.,60.,70.],'sex':[1.,0.,1.],'SBP':[120.,140.,160.],'TC':[5.,6.,7.],'HDLC':[1.2,1.5,1.],'waist':[85.,90.,100.],'smoking':[0.,1.,1.],'diabetes':[0.,1.,1.],'treated_hypertension':[0.,1.,1.],'family_history':[0.,1.,1.],'FBG':[5.,6.,7.],'eGFR':[100.,80.,60.]})
        expected={'Framingham':[.03840285314121217,.37571202642242585,.9560345861654868], 'CHINA-PAR':[.011523022105378589,.1741490576426329,.28477440159507617], 'CKM2S2-BAG':[3.,12.,27.]}
        for name,values in scores(frame).items():np.testing.assert_allclose(values,expected[name],rtol=1e-12,atol=1e-14)


if __name__=='__main__':unittest.main()
