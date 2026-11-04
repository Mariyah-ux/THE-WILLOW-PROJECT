from io import BytesIO
import json
import pandas as pd
import pytest
from BACKEND.context_review import review_reason, REVIEW
from test_pipeline import api

@pytest.mark.parametrize('text',[
 "I don't want to die",'I do not want to kill myself','I am not suicidal',
 'I am not thinking about suicide',"I don't want to hurt myself",'I will not harm myself',
 'I have no intention to end my life','I want to live','I wish to stay alive',
 'I want to live, but I want to die','I want to die. I do not want to die.',
 'I am not suicidal; however I plan to kill myself',
 'He said "I want to die"','She wrote: ‘I want to die’',
 'My friend said he wanted to commit suicide',
 'What if someone wanted to commit suicide?','Hypothetically, I want to die',
 'Suppose someone is suicidal','If I were to kill myself, what would happen?',
 'I DON’T want to die!',"I'm not suicidal",
])
def test_review(text):assert review_reason(text)

@pytest.mark.parametrize('text',[
 'I want to die','I want to kill myself',"I don't want to live",'I do not want to keep living',
 'I want to end my life','I am thinking about suicide',
 'I live in London','I want to live in London','I want to live with my sister',
 'The live concert was good','I am not hungry','Suicide is a word',
 'I do not want to eat. I want to die','I am not. Thinking about suicide is painful',
 'I am not only thinking about suicide','Nice weather',
 'What if it rains? I want to die','He said hello. I want to die',
])
def test_not_overbroad(text):assert review_reason(text) is None

def test_internal_diagnostics():
 from BACKEND import model_handler as mh
 r=mh.predict_text('I want to live');assert r['prediction']==REVIEW and r['model_score'] is None
 d=r['_model_diagnostics'];assert d['prediction'] in {'suicide','non-suicide'} and 0<=d['model_score']<=1
 x=mh.vectorizer.transform([mh.clean_text('I want to live')])[:,mh.selected_feature_indices]
 assert d['prediction']==mh.number_to_label[int(mh.model.predict(x)[0])]
 assert d['model_score']==float(mh.model.predict_proba(x)[0,list(mh.model.classes_).index(1)])

def test_api_review_and_history(api):
 _,client,db=api;r=client.post('/predict',json={'text':'I want to live'})
 assert r.status_code==200 and r.json['prediction']==REVIEW and r.json['model_score'] is None
 assert '_model_diagnostics' not in r.json and 'review_reason' not in r.json
 assert db.get_prediction_history()[0]['prediction']==REVIEW
 assert db.get_prediction_history()[0]['model_score'] is None
 h=client.get('/history');assert h.status_code==200 and REVIEW in h.get_data(as_text=True)

def test_mixed_csv(api):
 _,client,db=api
 r=client.post('/predict-csv',data={'file':(BytesIO(b'text\nI want to live\nI enjoyed a good book\n'),'probe.csv')})
 assert r.status_code==200
 d=pd.read_csv(BytesIO(r.data));assert d.prediction[0]==REVIEW and pd.isna(d.model_score[0])
 assert d.prediction[1] in {'suicide','non-suicide'} and 0<=d.model_score[1]<=1
 assert '_model_diagnostics' not in d.columns
 assert db.get_prediction_count()==2
