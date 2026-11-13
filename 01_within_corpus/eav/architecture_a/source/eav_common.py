#!/usr/bin/env python3
"""Shared, reproducible EAV 5-emotion, 42-subject trial-disjoint evaluation.

EAV reference: Lee et al., Scientific Data 2024, doi:10.1038/s41597-024-03838-4
Never split 5-s windows independently: split original 20-s speaking instances FIRST.
"""
from __future__ import annotations
import hashlib, json, os, re, warnings
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_recall_fscore_support,
                             roc_auc_score, roc_curve, precision_recall_curve, auc, log_loss,
                             balanced_accuracy_score, cohen_kappa_score, matthews_corrcoef)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

LABELS = ('Neutral', 'Sadness', 'Anger', 'Happiness', 'Calmness')
SLUGS = ('neutral', 'sadness', 'anger', 'happiness', 'calmness')
ALIASES = {'neutral':0, 'sad':1, 'sadness':1, 'angry':2, 'anger':2,
           'happy':3, 'happiness':3, 'calm':4, 'calmness':4, 'relax':4, 'relaxation':4}
N_CLASSES = 5
DEFAULT_DATA = Path('/workspace/data/EAV')
DEFAULT_OUTPUT = Path('/workspace/avyakt-sc-final/final-sc-eval-models/outputs/EAV_70_15_15')
PROTOCOL = 'EAV_42_SUBJECT_OVERLAPPING_ORIGINAL_SPEAKING_RECORDING_DISJOINT_STRATIFIED_70_15_15_SEED42'
SPLIT_SEED = 42


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4<<20), b''):h.update(block)
    return h.hexdigest()


def json_dump(path, obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    t=path.with_suffix(path.suffix+'.tmp')
    t.write_text(json.dumps(obj,indent=2,default=str,allow_nan=False)+'\n')
    os.replace(t,path)


def emotion_id(token):
    t=re.sub('[^a-z]','',str(token).lower())
    if t not in ALIASES:raise ValueError('Unrecognized EAV emotion: '+repr(token))
    return ALIASES[t]


def _onehot_labels(label_mat):
    m=sio.loadmat(label_mat)
    if 'label' not in m:raise ValueError(f'{label_mat}: missing expected MATLAB key label; keys={list(m)}')
    x=np.asarray(m['label'])
    if x.shape == (200,10):x=x.T
    if x.shape != (10,200):raise ValueError(f'{label_mat}: label must be 10x200 (or transpose), got {x.shape}')
    if not np.allclose(x.sum(0),1) or not np.all((x==0)|(x==1)):
        raise ValueError(f'{label_mat}: labels must be binary one-hot per original instance')
    return x.argmax(0).astype(int)


def _mat_metadata(p):
    try:
        fields=sio.whosmat(str(p))
        matches=[(name,shape) for name,shape,_ in fields if name in ('seg','seg1')]
    except NotImplementedError:
        matches=[]
    if not matches:
        # v7.3: HDF5 uses reverse axis order; label verification still uses scipy.io.
        raise ValueError(f'{p}: expected EEG MATLAB variable seg or seg1 in scipy-readable .mat')
    name,shape=matches[0]
    if len(shape)!=3 or sorted(shape)!=[30,200,10000]:
        raise ValueError(f'{p}: expected 200x10000x30 EEG array in some order, got {shape}')
    return name, shape


def _match_audio(name):
    # Publisher files, e.g. 002_Trial_02_Speaking_Neutral_Aud.wav
    pat=r'^(\d+)_Trial_(\d+)_(?:cond_)?Speaking_(Neutral|Sadness|Anger|Happiness|Calmness)_Aud\.wav$'
    m=re.match(pat,name,re.I)
    if not m:raise ValueError(f'Unexpected WAV filename: {name}; expected 002_Trial_02_Speaking_Neutral_Aud.wav')
    return int(m[1]),int(m[2]),emotion_id(m[3])


def build_manifest(data_root=DEFAULT_DATA, expected_subjects=tuple(range(1,43)), strict=True):
    """Audit the raw data without loading EEG signal matrices into memory.

    Cross-check every audio filename emotion against the corresponding EEG one-hot
    label at INSTANCE-1. This avoids guessing the meaning of 10 EEG label indices.
    """
    root=Path(data_root).expanduser().resolve()
    if not root.exists():raise FileNotFoundError(root)
    paths=sorted([x for x in root.iterdir() if x.is_dir() and re.fullmatch(r'subject\d+',x.name,re.I)],
                 key=lambda p:int(re.search(r'\d+',p.name).group()))
    sids={int(p.name.lower().removeprefix('subject')) for p in paths}
    expected=set(expected_subjects)
    if strict and sids!=expected:raise ValueError(f'EAV subject dirs mismatch; missing={sorted(expected-sids)} unexpected={sorted(sids-expected)}')
    records=[];mapping={};info=[]
    for p in paths:
        sid=int(p.name.lower().replace('subject',''))
        aud=p/'Audio';eegd=p/'EEG'
        if not aud.is_dir() or not eegd.is_dir():raise FileNotFoundError(f'Expected Audio/ and EEG/ in {p}')
        wavs=sorted([x for x in aud.iterdir() if x.suffix.lower()=='.wav'])
        eeg_mats=sorted(x for x in eegd.iterdir() if x.suffix.lower()=='.mat' and not x.stem.lower().endswith('_label'))
        label_mats=sorted(x for x in eegd.iterdir() if x.suffix.lower()=='.mat' and x.stem.lower().endswith('_label'))
        if len(eeg_mats)!=1 or len(label_mats)!=1:
            raise ValueError(f'{p}: expected 1 EEG .mat and 1 label .mat, got EEG={eeg_mats}, label={label_mats}')
        mat_key,mat_shape=_mat_metadata(eeg_mats[0]);codes=_onehot_labels(label_mats[0])
        if strict and len(wavs)!=100:raise ValueError(f'{p}: expected 100 speaking WAV files; found {len(wavs)}')
        seen=set();code_to_class={};class_to_code={};class_counter=Counter()
        for f in wavs:
            ix,trial,y=_match_audio(f.name)
            if not 1<=ix<=200 or ix in seen:raise ValueError(f'{f}: invalid/duplicate original instance {ix}')
            seen.add(ix);code=int(codes[ix-1]);class_counter[y]+=1
            if code in code_to_class and code_to_class[code]!=y:raise ValueError(f'EEG label index {code} has two emotions in {p}')
            if y in class_to_code and class_to_code[y]!=code:raise ValueError(f'{LABELS[y]} maps to two EEG label indices in {p}')
            code_to_class[code]=y;class_to_code[y]=code
            records.append({'recording_id':f's{sid:02d}_i{ix:03d}', 'subject_id':sid,
                            'trial_id':ix,'conversation_trial':trial,'emotion_id':y,
                            'emotion':LABELS[y], 'eeg_label_code':code, 'eeg_instance_zero':ix-1,
                            'audio_path':str(f.resolve()),'eeg_path':str(eeg_mats[0].resolve()),
                            'eeg_mat_key':mat_key,'eeg_shape':str(mat_shape),
                            'audio_bytes':f.stat().st_size,'audio_mtime_ns':f.stat().st_mtime_ns,
                            'eeg_bytes':eeg_mats[0].stat().st_size,'eeg_mtime_ns':eeg_mats[0].stat().st_mtime_ns})
        if strict and (len(code_to_class)!=5 or set(class_counter.values())!={20}):
            raise ValueError(f'{p}: expected 20 audio recordings per emotion; observed {dict(class_counter)}; label codes {code_to_class}')
        mapping[str(sid)]={LABELS[c]:int(k) for k,c in sorted(code_to_class.items())}
        info.append({'subject_id':sid,'audio_count':len(wavs),'mat_key':mat_key,'mat_shape':str(mat_shape),
                     'class_counts':{LABELS[c]:int(class_counter[c]) for c in range(5)},'label_codes':mapping[str(sid)]})
    df=pd.DataFrame(records)
    if df.empty:raise ValueError('Found no EAV speaking recordings')
    if df.recording_id.duplicated().any():raise ValueError('Duplicate original instance identity')
    if strict and len(df)!=4200:raise ValueError(f'Expected 4,200 original speaking recordings, found {len(df)}')
    # Per-subject + emotion exact count 20 -> split 14/3/3 without window leakage.
    split={}
    for (sid,y),g in df.groupby(['subject_id','emotion_id'],sort=True):
        keys=sorted(g.recording_id.tolist())
        rng=np.random.default_rng(SPLIT_SEED+sid*10007+y*997)
        order=rng.permutation(len(keys))
        n=len(keys)
        if strict and n!=20:raise ValueError(f'subject{sid} {LABELS[y]} expected 20, found {n}')
        ntr=round(.70*n);nval=round(.15*n)
        if ntr<1 or nval<1 or n-ntr-nval<1:raise ValueError('Too few trials for 70/15/15')
        for sp,indices in [('train',order[:ntr]),('val',order[ntr:ntr+nval]),('test',order[ntr+nval:])]:
            for i in indices:split[keys[i]]=sp
    df['split']=df.recording_id.map(split)
    df=df.sort_values(['subject_id','trial_id']).reset_index(drop=True)
    if df.split.isna().any():raise RuntimeError('Unassigned recordings')
    for sp in ('train','val','test'):
        grp=df[df.split==sp]
        if strict and set(grp.subject_id)!=set(range(1,43)):raise RuntimeError('Subjects missing in '+sp)
    return df, {'dataset_root':str(root),'protocol':PROTOCOL,'n_subjects':len(paths),
                'n_original_recordings':len(df),'labels':list(LABELS),'mat_metadata':info,
                'eeg_label_mapping_per_subject':mapping,
                'splits':{sp:int((df.split==sp).sum()) for sp in ('train','val','test')},
                'split_units':'original 20-second speaking recordings, not windows',
                'subject_overlap':True,'original_recording_overlap':False,'seed':SPLIT_SEED}


def manifest_sha(df):
    cols=['recording_id','subject_id','trial_id','emotion_id','eeg_label_code','split']
    return hashlib.sha256(df[cols].to_csv(index=False).encode()).hexdigest()


def audit_or_load(root, shared_dir, strict=True):
    """Use one canonical split for EEG and speech; never regenerate independently."""
    shared_dir=Path(shared_dir);shared_dir.mkdir(parents=True,exist_ok=True)
    df,summary=build_manifest(root,strict=strict)
    dest=shared_dir/'immutable_original_recording_split.csv'
    if dest.exists():
        old=pd.read_csv(dest)
        cols=['recording_id','subject_id','trial_id','emotion_id','eeg_label_code','split']
        if df[cols].to_csv(index=False)!=old[cols].to_csv(index=False):
            raise RuntimeError('IMMUTABLE SPLIT CONFLICT. Dataset/order changed; use a new output root.')
    else:df.to_csv(dest,index=False)
    summary['manifest_sha256']=manifest_sha(df)
    json_dump(shared_dir/'audit_summary.json',summary)
    df.groupby(['split','subject_id','emotion'],observed=True).size().rename('n_recordings').reset_index().to_csv(shared_dir/'subject_class_split_counts.csv',index=False)
    split_plot(shared_dir,df)
    return df,summary


def split_plot(out,df):
    fig,ax=plt.subplots(figsize=(8.5,5));wide=df.groupby(['emotion','split']).size().unstack(fill_value=0).reindex(LABELS)
    for split in ('train','val','test'):
        ax.bar(np.arange(5)+{'train':-.27,'val':0,'test':.27}[split],wide[split],width=.26,label=split)
    ax.set_xticks(range(5),LABELS);ax.set_ylabel('Original 20-second recordings')
    ax.set_title('EAV: stratified 70/15/15 split per emotion');ax.legend()
    fig.tight_layout();fig.savefig(out/'split_class_distribution.png',dpi=220);plt.close(fig)


def report_predictions(predictions, out, name='test', bootstrap=10000, epoch=None):
    """Original-recording metrics only. Saves CSV, JSON, PNG, SVG, and LaTeX tables."""
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    p=predictions.copy().sort_values(['subject_id','recording_id']).reset_index(drop=True)
    cols=[f'p_{slug}' for slug in SLUGS]
    if not {'recording_id','subject_id','emotion_id',*cols}.issubset(p.columns):
        raise ValueError('Bad predictions CSV; need ID, subject, true label, all probabilities')
    if p.recording_id.duplicated().any():raise ValueError('Duplicate original-recording predictions')
    probs=p[cols].to_numpy(dtype=float)
    if not np.isfinite(probs).all() or not np.allclose(probs.sum(axis=1),1,atol=1e-3):
        raise ValueError('Probabilities must be finite and sum to 1')
    y=p.emotion_id.to_numpy(dtype=int);pred=probs.argmax(axis=1)
    p['pred_class']=pred;p['pred_emotion']=[LABELS[z] for z in pred];p['correct']=(pred==y).astype(int)
    p['confidence']=probs.max(axis=1)
    p.to_csv(out/'recording_predictions.csv',index=False)
    subject=[];sc=[];subject_cm={}
    for sid,g in p.groupby('subject_id'):
        t=g.emotion_id.to_numpy(int);v=g.pred_class.to_numpy(int)
        prec,rec,f1,supp=precision_recall_fscore_support(t,v,labels=list(range(5)),zero_division=0)
        cm=confusion_matrix(t,v,labels=list(range(5)));subject_cm[int(sid)]=cm
        subject.append({'subject_id':int(sid),'n_recordings':len(g),'correct':int((t==v).sum()),
                        'accuracy':float((t==v).mean()),'macro_precision':float(prec.mean()),
                        'uar':float(rec.mean()),'macro_f1':float(f1.mean())})
        sc.extend({'subject_id':int(sid),'emotion':LABELS[i],'class_id':i,'precision':float(prec[i]),
                   'recall':float(rec[i]),'f1':float(f1[i]),'support':int(supp[i])} for i in range(5))
    subs=pd.DataFrame(subject);sclass=pd.DataFrame(sc)
    prec,rec,f1,supp=precision_recall_fscore_support(y,pred,labels=list(range(5)),zero_division=0)
    class_df=pd.DataFrame({'emotion':LABELS,'precision':prec,'recall':rec,'f1':f1,'support':supp})
    class_df['specificity']=[float((((y!=i)&(pred!=i)).sum())/max(1,(y!=i).sum())) for i in range(5)]
    class_df['auroc_ovr']=[float(roc_auc_score((y==i).astype(int),probs[:,i])) if 0<(y==i).sum()<len(y) else np.nan for i in range(5)]
    class_df['average_precision']=[float(__import__('sklearn.metrics',fromlist=['average_precision_score']).average_precision_score((y==i).astype(int),probs[:,i])) if (y==i).any() else np.nan for i in range(5)]
    cm=confusion_matrix(y,pred,labels=list(range(5)))
    rng=np.random.default_rng(2026)
    vs=subs.accuracy.to_numpy(float)
    boot=rng.choice(vs,size=(bootstrap,len(vs)),replace=True).mean(axis=1) if bootstrap else np.array([vs.mean()])
    metrics={'split':name,'protocol':PROTOCOL,'n_subjects':len(subs),'n_original_recordings':len(p),
             'mean_subject_accuracy':float(vs.mean()), 'subject_accuracy_std':float(vs.std(ddof=1)) if len(vs)>1 else 0.,
             'subject_bootstrap_95ci':list(map(float,np.percentile(boot,[2.5,97.5]))),
             'pooled_recording_accuracy':float(accuracy_score(y,pred)),
             'macro_precision':float(prec.mean()),'macro_recall':float(rec.mean()),
             'uar':float(rec.mean()), 'macro_f1':float(f1.mean()),
             'weighted_f1':float(np.average(f1,weights=supp)),
             'mean_subject_macro_f1':float(subs.macro_f1.mean()),
             'cohen_kappa':float(cohen_kappa_score(y,pred)),
             'mcc':float(matthews_corrcoef(y,pred)),
             'log_loss':float(log_loss(y,probs,labels=list(range(5)))),
             'multiclass_brier':float(np.mean(np.sum((probs-np.eye(5)[y])**2,axis=1))),
             'chance_accuracy':.20,'best_epoch':epoch}
    try:metrics['auroc_macro_ovr']=float(roc_auc_score(y,probs,labels=list(range(5)),multi_class='ovr',average='macro'))
    except ValueError:metrics['auroc_macro_ovr']=None
    subs.to_csv(out/'per_subject_metrics.csv',index=False)
    sclass.to_csv(out/'per_subject_class_metrics.csv',index=False)
    class_df.to_csv(out/'per_class_metrics.csv',index=False)
    pd.DataFrame(cm,index=LABELS,columns=LABELS).to_csv(out/'confusion_counts.csv')
    pd.DataFrame(cm/np.maximum(cm.sum(axis=1,keepdims=True),1),index=LABELS,columns=LABELS).to_csv(out/'confusion_normalized.csv')
    json_dump(out/'metrics.json',metrics)
    for filename,frame in [('table_subjects.tex',subs),('table_emotions.tex',class_df)]:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore',FutureWarning)
            (out/filename).write_text(frame.to_latex(index=False,float_format='%.4f'))
    (out/'table_summary.tex').write_text(pd.DataFrame([metrics]).drop(columns=['subject_bootstrap_95ci','protocol']).to_latex(index=False,float_format='%.4f'))
    _plot_report(out,name,p,subs,sclass,class_df,cm,subject_cm,metrics,probs,y)
    return metrics


def _heatmap(matrix,xlabels,ylabels,title,out,percent=False,vector=True):
    fig,ax=plt.subplots(figsize=(max(7,len(xlabels)*1.25),max(4.8,len(ylabels)*.23+2)))
    im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1 if percent else None,aspect='auto')
    if len(ylabels)<=6:
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                v=matrix[i,j];ax.text(j,i,f'{v:.2f}' if percent else str(int(v)),ha='center',va='center',fontsize=10,color='white' if im.norm(v)>.65 else 'black')
    ax.set_xticks(range(len(xlabels)),xlabels,rotation=25,ha='right')
    ax.set_yticks(range(len(ylabels)),ylabels)
    ax.set_xlabel('Predicted emotion' if len(ylabels)==5 else 'Emotion')
    ax.set_ylabel('True emotion' if len(ylabels)==5 else 'Subject')
    ax.set_title(title);fig.colorbar(im,ax=ax,shrink=.8);fig.tight_layout()
    fig.savefig(out.with_suffix('.png'),dpi=145 if not vector else 200)
    if vector:fig.savefig(out.with_suffix('.svg'))
    plt.close(fig)


def _plot_report(out,name,p,subs,sc,byclass,cm,subject_cm,met,probs,y):
    _heatmap(cm,LABELS,LABELS,f'{name}: confusion matrix — 20-second recordings',out/'confusion_counts')
    _heatmap(cm/np.maximum(cm.sum(1,keepdims=True),1),LABELS,LABELS,f'{name}: normalized confusion',out/'confusion_normalized',True)
    xs=np.arange(len(subs));fig,ax=plt.subplots(figsize=(15,5))
    ax.bar(xs,subs.accuracy.to_numpy()*100,width=.8)
    ax.axhline(met['mean_subject_accuracy']*100,ls='--',color='black',label=f'Mean {met["mean_subject_accuracy"]*100:.2f}%')
    ax.set_xticks(xs,[f'S{int(s):02d}' for s in subs.subject_id],rotation=65)
    ax.set_ylabel('Accuracy (%)');ax.set_ylim(0,105);ax.set_title(f'{name}: individual subjects (n={len(subs)})');ax.legend();fig.tight_layout()
    fig.savefig(out/'subject_accuracy.png',dpi=200);fig.savefig(out/'subject_accuracy.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.7));z=np.arange(5);w=.25
    for k,key in enumerate(('precision','recall','f1')):ax.bar(z+(k-1)*w,byclass[key].to_numpy()*100,width=w,label=key.title())
    ax.set_xticks(z,LABELS);ax.set_ylim(0,105);ax.set_ylabel('%');ax.legend();ax.set_title(f'{name}: per-emotion performance');fig.tight_layout()
    fig.savefig(out/'classwise_precision_recall_f1.png',dpi=200);fig.savefig(out/'classwise_precision_recall_f1.svg');plt.close(fig)
    labs=[f'S{int(z):02d}' for z in subs.subject_id]
    for metric in ('precision','recall','f1'):
        mat=sc.pivot(index='subject_id',columns='class_id',values=metric).loc[subs.subject_id,range(5)].to_numpy()
        _heatmap(mat,LABELS,labs,f'{name}: subject × emotion {metric}',out/f'subject_emotion_{metric}',True)
    indiv=out/'individual_subject_confusions';indiv.mkdir(exist_ok=True)
    for sid,m in subject_cm.items():
        _heatmap(m,LABELS,LABELS,f'Subject {sid} — {name}',indiv/f'subject{sid:02d}_confusion',vector=False)
    fig,ax=plt.subplots(figsize=(7,6))
    for c in range(5):
        binary=(y==c).astype(int)
        if binary.sum() and binary.sum()<len(binary):
            fpr,tpr,_=roc_curve(binary,probs[:,c]);ax.plot(fpr,tpr,label=f'{LABELS[c]} (AUC={auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1],linestyle='--',color='gray');ax.legend();ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{name}: OvR ROC');fig.tight_layout()
    fig.savefig(out/'roc_ovr.png',dpi=200);fig.savefig(out/'roc_ovr.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,6))
    for c in range(5):
        pr,rc,_=precision_recall_curve((y==c).astype(int),probs[:,c]);ax.plot(rc,pr,label=LABELS[c])
    ax.legend();ax.set(xlabel='Recall',ylabel='Precision',title=f'{name}: OvR PR curves');fig.tight_layout()
    fig.savefig(out/'precision_recall_curves.png',dpi=200);fig.savefig(out/'precision_recall_curves.svg');plt.close(fig)
    conf=probs.max(1);correct=(probs.argmax(1)==y)
    fig,ax=plt.subplots(figsize=(8,4.5));ax.hist(conf[correct],bins=20,alpha=.7,label='Correct');ax.hist(conf[~correct],bins=20,alpha=.7,label='Incorrect')
    ax.set(xlabel='Maximum class probability',ylabel='Count',title=f'{name}: confidence histogram');ax.legend();fig.tight_layout()
    fig.savefig(out/'confidence_distribution.png',dpi=200);plt.close(fig)
    edges=np.linspace(0,1,11);centers=[];acc=[];mass=[]
    for a,b in zip(edges[:-1],edges[1:]):
        ix=(conf>=a)&(conf<=b if b==1 else conf<b)
        if ix.any():centers.append(float(conf[ix].mean()));acc.append(float(correct[ix].mean()));mass.append(int(ix.sum()))
    pd.DataFrame({'mean_confidence':centers,'accuracy':acc,'support':mass}).to_csv(out/'reliability_bins.csv',index=False)
    fig,ax=plt.subplots(figsize=(5.8,5.8));ax.plot([0,1],[0,1],'--',color='gray');ax.plot(centers,acc,'o-')
    ax.set(xlabel='Confidence',ylabel='Empirical accuracy',title=f'{name}: reliability');fig.tight_layout()
    fig.savefig(out/'reliability_diagram.png',dpi=200);plt.close(fig)
