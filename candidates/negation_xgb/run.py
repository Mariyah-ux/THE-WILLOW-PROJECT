"""One-candidate reproduction: train/validate first, then one explicit final test evaluation."""
import argparse, hashlib, json, time, sys
from pathlib import Path
from importlib.metadata import version
import joblib, numpy as np, pandas as pd
from scipy.sparse import load_npz
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from xgboost import XGBClassifier
from preprocessing import clean_text, _nlp_tools
import re
EXPECTED='1ec16b065a2fae36488de16e8abfc326586a576a25901e09467223e545ddd295'
HERE=Path(__file__).resolve().parent

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def write(p,d):p.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n')
def metrics(y,p):
 pr,rc,f,_=precision_recall_fscore_support(y,p,average='macro',zero_division=0)
 sp,sr,sf,_=precision_recall_fscore_support(y,p,labels=[1],average=None,zero_division=0)
 cm=confusion_matrix(y,p,labels=[0,1])
 return dict(accuracy=float(accuracy_score(y,p)),macro_precision=float(pr),macro_recall=float(rc),macro_f1=float(f),suicide_precision=float(sp[0]),suicide_recall=float(sr[0]),suicide_f1=float(sf[0]),confusion_matrix=cm.tolist(),false_negatives=int(cm[1,0]),false_positives=int(cm[0,1]))
def legacy_clean(text):
 from nltk.corpus import stopwords
 from nltk import word_tokenize
 words,lemma=_nlp_tools()
 historical=set(stopwords.words('english'))-{'not','no','never','nor',"don't","can't","won't"}
 t=text.lower();t=re.sub(r'http\S+|www\S+','',t);t=re.sub(r'<.*?>','',t);t=re.sub(r'[^a-zA-Z\s]',' ',t)
 return ' '.join(lemma.lemmatize(w) for w in word_tokenize(t) if w not in historical)
def main():
 a=argparse.ArgumentParser();a.add_argument('stage',choices=['train','finalize']);a.add_argument('--dataset',required=True);a.add_argument('--legacy-root',required=True);args=a.parse_args()
 data=Path(args.dataset);root=Path(args.legacy_root)
 assert data.stat().st_size==166902029 and sha(data)==EXPECTED,'Dataset integrity failed'
 d=pd.read_csv(data)
 if args.stage=='train':
  assert not (HERE/'model.pkl').exists(),'Refusing to overwrite existing candidate'
  originals={str(p.relative_to(root)):sha(p) for folder in ['saved_models','Feature_Artifacts','GA_Artifacts'] for p in sorted((root/folder).iterdir()) if p.is_file() and not p.name.startswith('.')}
  assert len(originals)==20;write(HERE/'original_hashes.json',originals)
  raw=len(d);d=d.dropna(subset=['text','class']).drop_duplicates(subset=['text'])
  print('Recovering historical eligibility and split',flush=True)
  old=[]
  for i,t in enumerate(d.text):
   old.append(legacy_clean(t))
   if i%50000==0:print('Historical rows',i,flush=True)
  old=pd.Series(old,index=d.index);keep=old.str.strip().ne('');d=d.loc[keep];old=old.loc[keep]
  assert set(d['class'])=={'suicide','non-suicide'}
  tr,tmp=train_test_split(d.index.to_numpy(),test_size=.2,random_state=42,stratify=d['class'])
  va,te=train_test_split(tmp,test_size=.5,random_state=42,stratify=d.loc[tmp,'class'])
  split={'train':tr,'val':va,'test':te};checks={}
  for n,r in split.items():
   checks[n]=np.array_equal(d.loc[r,'class'].to_numpy(),np.load(root/f'Feature_Artifacts/y_{n}.npy',allow_pickle=True))
  assert all(checks.values()),'Split label sequence mismatch; no training'
  original_v=joblib.load(root/'Feature_Artifacts/tfidf_vectorizer.pkl');alignment={}
  for n,r in [('train',tr),('val',va)]:
   saved=load_npz(root/f'Feature_Artifacts/X_{n}.npz');actual=original_v.transform(old.loc[r]);delta=saved-actual
   error=float(np.max(np.abs(delta.data))) if delta.nnz else 0.;alignment[n]=error
   assert saved.shape==actual.shape and error<1e-6,'Historical matrix alignment mismatch'
   del saved,actual,delta
  np.savez_compressed(HERE/'split_indices.npz',**split)
  write(HERE/'split_metadata.json',dict(dataset_sha256=EXPECTED,raw_rows=raw,eligible_rows=len(d),removed_rows=raw-len(d),seed=42,proportions=[.8,.1,.1],sizes={n:len(r) for n,r in split.items()},saved_label_order_matches=checks,historical_matrix_max_absolute_error=alignment,row_identity='zero-based CSV data row, not an invented original ID'))
  print('Split verified. Preparing corrected training and validation text.',flush=True)
  corrected={n:[clean_text(t) for t in d.loc[r,'text']] for n,r in [('train',tr),('val',va)]}
  v=TfidfVectorizer(max_features=5000,ngram_range=(1,2),min_df=3,max_df=.95,sublinear_tf=True,dtype=np.float32)
  x=v.fit_transform(corrected['train']);xv=v.transform(corrected['val']);print('Sparse matrices ready',x.shape,xv.shape,flush=True)
  old_model=joblib.load(root/'saved_models/XGBoost.pkl');params=old_model.get_params();m=XGBClassifier(**params)
  y=(d.loc[tr,'class']=='suicide').to_numpy(dtype=np.int32);yv=(d.loc[va,'class']=='suicide').to_numpy(dtype=np.int32)
  start=time.perf_counter();m.fit(x,y);elapsed=time.perf_counter()-start
  joblib.dump(m,HERE/'model.pkl');joblib.dump(v,HERE/'vectorizer.pkl');joblib.dump({'non-suicide':0,'suicide':1},HERE/'label_mapping.pkl')
  versions={p:version(p) for p in ['numpy','scipy','pandas','scikit-learn','xgboost','joblib','nltk']}
  versions['python']=sys.version.split()[0]
  write(HERE/'artifact_manifest.json',dict(status='new reproduction',features=len(v.get_feature_names_out()),versions=versions,sha256={p:sha(HERE/p) for p in ['model.pkl','vectorizer.pkl','label_mapping.pkl','preprocessing.py']}))
  write(HERE/'configuration.json',dict(status='new reproduction from recorded implementation',xgboost={k:('NaN' if isinstance(z,float) and np.isnan(z) else z) for k,z in params.items()},tfidf=dict(max_features=5000,ngram_range=[1,2],min_df=3,max_df=.95,sublinear_tf=True,dtype='float32'),versions=versions,preprocessing_sha256=sha(HERE/'preprocessing.py'),fit_seconds=elapsed,seed=42,threshold=.5,deviations=['Cleaner reconstructed from recorded code; SHA matches historical cleaner.','Split indices regenerated and checked against saved historical train/validation features.','Artifact files are new serializations, not recovered original bytes.']))
  result=dict(status='new reproduction',validation=metrics(yv,m.predict(xv)),historical_model_validation=metrics(yv,old_model.predict(load_npz(root/'GA_Artifacts/X_val_selected.npz'))),fit_seconds=elapsed,frozen=True,test_evaluations=0)
  write(HERE/'metrics.json',result)
  assert originals=={p:sha(root/p) for p in originals}
  print(json.dumps(result,indent=2),flush=True)
 else:
  result=json.loads((HERE/'metrics.json').read_text());assert result['frozen'] and result['test_evaluations']==0
  from smoke import load_bundle
  m,v,mapping=load_bundle();rows=np.load(HERE/'split_indices.npz')['test']
  result['test_evaluations']=1;write(HERE/'metrics.json',result)
  x=v.transform([clean_text(t) for t in d.loc[rows,'text']]);y=(d.loc[rows,'class']=='suicide').to_numpy(dtype=np.int32)
  result['test']=metrics(y,m.predict(x));write(HERE/'metrics.json',result)
  originals=json.loads((HERE/'original_hashes.json').read_text());assert all(sha(root/p)==h for p,h in originals.items())
  print(json.dumps(result['test'],indent=2),flush=True)
if __name__=='__main__':main()
