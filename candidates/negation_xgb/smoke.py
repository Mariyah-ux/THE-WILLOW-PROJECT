"""Load/infer with the exact packaged cleaner; no Flask or database side effects."""
import hashlib,json,sys,subprocess
from pathlib import Path
from importlib.metadata import version
import joblib
import numpy as np
from preprocessing import clean_text
HERE=Path(__file__).resolve().parent

def load_bundle():
 manifest=json.loads((HERE/'artifact_manifest.json').read_text())
 for p,h in manifest['sha256'].items():
  if hashlib.sha256((HERE/p).read_bytes()).hexdigest()!=h:raise RuntimeError('Artifact integrity mismatch: '+p)
 for name,v in manifest['versions'].items():
  if name!='python' and version(name)!=v:raise RuntimeError('Version mismatch: '+name)
 m=joblib.load(HERE/'model.pkl');v=joblib.load(HERE/'vectorizer.pkl');mapping=joblib.load(HERE/'label_mapping.pkl')
 assert mapping=={'non-suicide':0,'suicide':1}
 assert m.n_features_in_==len(v.get_feature_names_out())==5000
 assert set(m.classes_)=={0,1}
 return m,v,mapping

def predict(texts,bundle):
 if not isinstance(texts,list) or not texts:raise ValueError('Provide a non-empty list of texts')
 cleaned=[]
 for t in texts:
  if not isinstance(t,str) or not t.strip():raise ValueError('Non-empty text required')
  c=clean_text(t)
  if not c.strip():raise ValueError('No usable text remains')
  cleaned.append(c)
 m,v,mapping=bundle;x=v.transform(cleaned);labels=m.predict(x);scores=m.predict_proba(x)[:,list(m.classes_).index(1)]
 inverse={v:k for k,v in mapping.items()}
 return [dict(prediction=inverse[int(p)],model_score=float(s),cleaned_text=c) for p,s,c in zip(labels,scores,cleaned)]

PROBES=['I want to die',"I don't want to die",'I want to live',"I don't want to live",'I am thinking about suicide','I am not thinking about suicide','I want to hurt myself',"I don't want to hurt myself",'I want to end my life','I do not want to end my life','I enjoyed a delicious pizza with my friends today.','I am not suicidal, but I want to end my life','The character said "I want to die"','Hypothetically, what if someone wanted to commit suicide?','My friend is thinking about suicide','Suicide prevention hotlines help people stay alive']
def main():
 bundle=load_bundle();out=predict(PROBES,bundle)
 if '--child' in sys.argv:print(json.dumps(out));return
 assert {r['prediction'] for r in out}=={'suicide','non-suicide'}
 assert out==predict(PROBES,bundle)
 assert all(predict([t],bundle)[0]==r for t,r in zip(PROBES,out))
 invalid=[[],[''],['123 !!!'],[None],['hello',42],None]
 for t in invalid:
  try:predict(t,bundle)
  except ValueError:pass
  else:raise AssertionError('Invalid input accepted')
 child=subprocess.run([sys.executable,'-B',str(HERE/'smoke.py'),'--child'],check=True,capture_output=True,text=True)
 assert json.loads(child.stdout)==out
 print(json.dumps(dict(status='PASS',checks=['saved bundle hashes and versions','model/vectorizer load and 5000 feature dimensions','binary label mapping','new input preprocessing','both output labels','batch matches single predictions','six invalid input cases','repeat determinism','clean subprocess determinism'],qualitative_probes=[dict(text=t,**r) for t,r in zip(PROBES,out)]),indent=2))
if __name__=='__main__':main()
