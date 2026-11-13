#!/usr/bin/env python3
"""PME4 Architecture A — 70/15/15 original-recording-disjoint splits, all 11 speakers.

Same exact WavLM-Large + GHF + DPTP + SupCon architecture/training strategy
as EAV Architecture A V3 (pretrained WavLM weights, training FROM SCRATCH on PME4).

7 PME4 emotions -> retain only Neutral=0, Sad=1, Angry=2, Happy=3.
Labels read exclusively from PME4_dataset_configs.csv. All 11 subjects appear in
all three partitions. Stratify by (subject, emotion), assigning each distinct
(subject, trial) recording to exactly one split. PME4 trial indices may have
DIFFERENT emotions across subjects, so global trial-ID grouping is invalid.
For 50 trials/class/subject: 35 train + 7/8 val + 8/7 test (balanced globally)
=> 70/15/15 across the 4 retained classes and across the entire dataset.

CRITICAL: speaker-overlapping evaluation; NOT subject-independent. Results cannot
be compared as if they were from 7/2/2 disjoint subjects. Emotion/class and
trial position may still be associated. Select best on validation only, then
perform an explicit ONE-TIME final test using --mode test --unlock-test.

Usage: --mode audit ; --mode train [--resume] ; --mode test --unlock-test
"""
from __future__ import annotations
import argparse, hashlib, json, logging, math, os, platform, random, re, shutil, sys, time, traceback
from collections import Counter, defaultdict
from pathlib import Path

def parse_args():
    p=argparse.ArgumentParser(description='PME4 | Architecture A | all 11 subjects with subject-class stratified 70/15/15 split')
    p.add_argument('--mode',choices=('audit','train','test'),default='train')
    p.add_argument('--data-root',type=Path,default=Path('/workspace/avyakt-sc/pme4-dataset-extracted/pme4-dataset'))
    p.add_argument('--project-root',type=Path,default=Path('/workspace/avyakt-sc-final'))
    p.add_argument('--output-dir',type=Path,default=None)
    p.add_argument('--batch-size',type=int,choices=[4,8],default=8)
    p.add_argument('--grad-accum',type=int,default=None)
    p.add_argument('--eval-batch-size',type=int,default=8)
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--max-epochs',type=int,default=28)
    p.add_argument('--frozen-epochs',type=int,default=4)
    p.add_argument('--patience',type=int,default=7)
    p.add_argument('--min-delta',type=float,default=.002)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--checkpoint-revision',default='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--unlock-test',action='store_true')
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--no-install',action='store_true',help='Dependencies must already be installed in your .venv')
    p.add_argument('--skip-wav-hash',action='store_true',help='NOT RECOMMENDED. Skip bytewise WAV hashing during audit.')
    p.add_argument('--export-train-embeddings',action='store_true')
    a=p.parse_args()
    if a.grad_accum is None:a.grad_accum=2 if a.batch_size==8 else 4
    if a.grad_accum*a.batch_size!=16:p.error('Effective batch must be 16: 8x2 or 4x4')
    if a.max_epochs<=a.frozen_epochs or a.frozen_epochs<1:p.error('Need 1<=frozen-epochs<max-epochs')
    if a.patience<1 or a.min_delta<0:p.error('Invalid early stopping parameters')
    if a.eval_batch_size<1 or a.workers<0:p.error('Invalid DataLoader configuration')
    if a.resume and a.mode!='train':p.error('--resume only valid for train')
    if a.unlock_test and a.mode!='test':p.error('--unlock-test only valid for test')
    return a

def install_missing(a):
    import importlib.util,subprocess
    deps={'torch':None,'numpy':'numpy>=1.24','pandas':'pandas>=2.0',
          'scipy':'scipy>=1.11','soundfile':'soundfile>=0.12',
          'sklearn':'scikit-learn>=1.3','tqdm':'tqdm>=4.65',
          'matplotlib':'matplotlib>=3.7','transformers':'transformers==4.46.3',
          'huggingface_hub':'huggingface_hub>=0.26,<1','safetensors':'safetensors>=0.4'}
    if a.mode=='audit':
        for k in ['transformers','huggingface_hub','safetensors']:deps.pop(k)
    missing=[(k,v) for k,v in deps.items() if importlib.util.find_spec(k) is None]
    if any(k=='torch' for k,v in missing):raise RuntimeError('Missing GPU PyTorch; install separately')
    if missing and a.no_install:raise RuntimeError('Missing Python packages: '+str(missing))
    if missing:subprocess.check_call([sys.executable,'-m','pip','install',*[v for k,v in missing]])

def make_logger(out):
    from tqdm.auto import tqdm
    out.mkdir(parents=True,exist_ok=True)
    lg=logging.getLogger('PME4_A');lg.setLevel(logging.INFO);lg.handlers.clear();lg.propagate=False
    form=logging.Formatter('%(asctime)s | %(levelname)-7s | %(message)s',datefmt='%Y-%m-%d %H:%M:%S')
    class Console(logging.Handler):
        def emit(self,rec):
            try:tqdm.write(self.format(rec),file=sys.stderr)
            except Exception:print(self.format(rec),file=sys.stderr)
    for h in [Console(),logging.FileHandler(out/'run.log')]:
        h.setFormatter(form);lg.addHandler(h)
    return lg


def main():
    import numpy as np
    import pandas as pd
    import soundfile as sf
    import scipy
    from scipy.signal import resample_poly
    from scipy.special import softmax
    from sklearn.metrics import (confusion_matrix,precision_recall_fscore_support,
                                 roc_auc_score,log_loss,roc_curve,auc)
    from tqdm.auto import tqdm
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.utils.data import Dataset,DataLoader,Sampler
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    args=ARGS
    DATA_ROOT=args.data_root.expanduser().resolve()
    PROJECT_ROOT=args.project_root.expanduser().resolve()
    OUTPUT_ROOT=(args.output_dir or PROJECT_ROOT/'outputs'/'pme4_architecture_a_all11_trial_70_15_15_v2_fixed').expanduser().resolve()
    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    (OUTPUT_ROOT/'audit').mkdir(exist_ok=True)
    (OUTPUT_ROOT/'audio_cache').mkdir(exist_ok=True)
    LOG=make_logger(OUTPUT_ROOT)
    def log(*s):LOG.info(' '.join(map(str,s)))
    SEED=args.seed; SAMPLE_RATE=16000;WINDOW_SAMPLES=64000
    NUM_EPOCHS=args.max_epochs;MICROBATCH_SIZE=args.batch_size
    EVAL_BATCH_SIZE=args.eval_batch_size;GRAD_ACCUM=args.grad_accum;NUM_WORKERS=args.workers
    CACHE_RESAMPLED_AUDIO=not args.no_cache
    DEVICE=torch.device('cuda:0')
    CHECKPOINT_NAME='microsoft/wavlm-large'
    CHECKPOINT_REVISION=args.checkpoint_revision
    TRAIN_SUBJECTS=VAL_SUBJECTS=TEST_SUBJECTS=set(range(1,12))
    CONFIG={'protocol_id':'PME4_SPEECH4_ALL11_SUBJECT_CLASS_STRATIFIED_70_15_15_V2',
            'name':'PME4_ARCH_A_WAVLM_LARGE_GHF_DPTP',
            'split':{sp:list(range(1,12)) for sp in ('train','val','test')},
            'seed':SEED,'fixed_split_seed':42,
            'source_script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dataset_root':str(DATA_ROOT),'window_samples':WINDOW_SAMPLES,
            'checkpoint_revision':CHECKPOINT_REVISION,
            'schedule':'4 frozen + up to 24 finetune; select full-val MSA',
            'target_classes':['Neutral','Sad','Angry','Happy'],
            'ignored_classes':['Fear','Disgust','Surprise'],
            'split_protocol':'Original-recording-disjoint, stratified by subject and emotion; all 11 subjects in train/val/test; 70/15/15',
            'caveat':'SPEAKER OVERLAP is intentional. This does not measure generalization to unseen subjects. '
                     'The same numeric trial ID may occur in different splits for different subjects; experimental order remains a potential shortcut.'}
    CONFIG['split_definition_sha256']='PENDING_DATASET_AUDIT'
    LABEL_NAMES=['Neutral','Sad','Angry','Happy']
    LABEL_SLUGS=['neutral','sad','angry','happy']
    LABEL_ID={'neutral':0,'sad':1,'angry':2,'happy':3}
    assert TRAIN_SUBJECTS==VAL_SUBJECTS==TEST_SUBJECTS==set(range(1,12))
    if not DATA_ROOT.is_dir():raise FileNotFoundError(f'PME4 root missing: {DATA_ROOT}')
    if args.mode!='audit' and not torch.cuda.is_available():raise RuntimeError('NVIDIA GPU required for train/test')
    if args.mode!='audit' and not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable GPU required')
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
    random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(SEED)
    log('PME4 Architecture A | ALL 11 subjects in train/val/test | subject-emotion stratified 70/15/15')
    log('DATA_ROOT:',DATA_ROOT,'OUTPUT:',OUTPUT_ROOT)
    log('All 11 subjects occur in each of train/val/test; subjects overlap BY DESIGN')
    log('Identical GHF/DPTP architecture; speaker-overlapping result is NOT subject independent')

    def save_json(path, payload):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(payload,indent=2,default=str)+'\n')
        os.replace(tmp,path)

    def sha256_file(path):
        h=hashlib.sha256()
        with open(path,'rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        return h.hexdigest()

    def norm_subject(x):
        # Metadata may encode subject as numeric 1 or string s01.
        s=str(x).strip().lower().replace('.0','')
        m=re.fullmatch(r'(?:s|subject)?0*(\d+)',s)
        if not m:raise ValueError('Cannot parse PME4 subject from '+repr(x))
        return int(m.group(1))
    def norm_trial(x):
        s=str(x).strip().lower().replace('.0','')
        m=re.fullmatch(r't?0*(\d+)',s)
        if not m:raise ValueError('Cannot parse PME4 trial from '+repr(x))
        return int(m.group(1))
    def norm_emotion(x):
        s=re.sub(r'[^a-z]','',str(x).strip().lower())
        return {'neutral':'neutral','sad':'sad','sadness':'sad',
                'angry':'angry','anger':'angry','happy':'happy','happiness':'happy',
                'fear':'fear','fearful':'fear','disgust':'disgust','disgusted':'disgust',
                'surprise':'surprise','surprised':'surprise'}.get(s)

    def audit_pme4():
        csv_path=DATA_ROOT/'PME4_dataset_configs.csv'
        if not csv_path.is_file():raise FileNotFoundError('Official PME4 metadata CSV missing: '+str(csv_path))
        raw=pd.read_csv(csv_path,low_memory=False)
        required={'subject','trial','emotion'}
        if not required.issubset(raw.columns):
            raise ValueError(f'PME4 metadata missing {required-set(raw.columns)}; columns={list(raw.columns)}')
        parsed=[];skipped=Counter();observed=set();by_subject_class=defaultdict(list)
        for i,r in raw.iterrows():
            sid=norm_subject(r['subject']);tid=norm_trial(r['trial']);emotion=norm_emotion(r['emotion'])
            if not 1<=sid<=11:raise ValueError(f'Unexpected PME4 speaker {sid} at CSV row {i}')
            if emotion is None:raise ValueError(f'Unknown emotion {r["emotion"]!r} in PME4 CSV row {i}')
            if (sid,tid) in observed:raise ValueError(f'Duplicate metadata identity s{sid:02d}_t{tid:03d}')
            observed.add((sid,tid))
            if emotion not in LABEL_ID:
                skipped[emotion]+=1;continue
            c=LABEL_ID[emotion]
            parsed.append((sid,tid,c,str(r['emotion'])))
            by_subject_class[(sid,c)].append(tid)
        if not parsed:raise RuntimeError('No retained four-class PME4 records')
        expected={(sid,c) for sid in range(1,12) for c in range(4)}
        if set(by_subject_class)!=expected:
            raise RuntimeError('Each of the 11 speakers must have four retained emotion classes; missing: '
                               +str(sorted(expected-set(by_subject_class))))

        # PME4 has participant-specific emotion order: t002 can be one emotion for
        # s01 and a different emotion for s02. Therefore DO NOT group trials by
        # numeric trial ID across speakers. Each (subject ID, trial ID) is a
        # unique original recording. Allocate its partition exactly once.
        split_for_recording={}
        subject_class_quota={}
        base_rng=np.random.default_rng(42)  # Fixed, independent of training seed.

        def apportion(class_counts, fraction, target, capacity):
            """Largest-remainder stratified allocation with global subject totals."""
            quotas={c:int(math.floor(float(n)*fraction)) for c,n in class_counts.items()}
            if any(quotas[c]>capacity[c] for c in quotas):
                raise RuntimeError('Not enough per-class capacity to apportion split')
            remaining=int(target)-sum(quotas.values())
            if remaining<0:
                raise RuntimeError(f'Impossible stratified split: {remaining} slots')
            # Randomized tie break is seeded and independent of input row order.
            order=list(class_counts)
            base_rng.shuffle(order)
            order.sort(key=lambda c: float(class_counts[c])*fraction - quotas[c],reverse=True)
            for c in order:
                if remaining==0:break
                if quotas[c]<capacity[c]:
                    quotas[c]+=1;remaining-=1
            if remaining:
                raise RuntimeError(f'Unable to allocate {remaining} split slots')
            return quotas

        for sid in range(1,12):
            class_counts={c:len(by_subject_class[(sid,c)]) for c in range(4)}
            if min(class_counts.values())<15:
                raise RuntimeError(f'Speaker s{sid:02d}: too few samples per emotion for 70/15/15: {class_counts}')
            n_subject=sum(class_counts.values())
            train_target=round(.70*n_subject)
            val_target=round(.15*n_subject)
            tr=apportion(class_counts,.70,train_target,{c:class_counts[c]-4 for c in range(4)})
            va=apportion(class_counts,.15,val_target,{c:class_counts[c]-tr[c]-2 for c in range(4)})
            te={c:class_counts[c]-tr[c]-va[c] for c in range(4)}
            if any(min(tr[c],va[c],te[c])<2 for c in range(4)):
                raise RuntimeError(f'Speaker s{sid:02d}: insufficient per-class partition count')
            assert sum(tr.values())==train_target
            assert sum(va.values())==val_target
            assert sum(te.values())==n_subject-train_target-val_target
            subject_class_quota[f's{sid:02d}']={LABEL_NAMES[c]:
                {'total':class_counts[c],'train':tr[c],'val':va[c],'test':te[c]} for c in range(4)}
            for c in range(4):
                tids=sorted(by_subject_class[(sid,c)])
                rng=np.random.default_rng(42 + sid*10_007 + c*997)
                rng.shuffle(tids)
                for sp, part in (
                    ('train',tids[:tr[c]]),
                    ('val',tids[tr[c]:tr[c]+va[c]]),
                    ('test',tids[tr[c]+va[c]:])):
                    for tid in part:
                        key=(sid,tid)
                        if key in split_for_recording:
                            raise RuntimeError(f'Duplicate recording partition assignment: {key}')
                        split_for_recording[key]=sp
        assert len(split_for_recording)==len(parsed)
        counts_global=Counter(split_for_recording.values())
        N=len(split_for_recording)
        fractions={sp:float(counts_global[sp]/N) for sp in ('train','val','test')}
        if abs(fractions['train']-.70)>.006 or abs(fractions['val']-.15)>.006 or abs(fractions['test']-.15)>.006:
            raise RuntimeError(f'Unable to achieve approximately 70/15/15 original-recording split: {fractions}')
        CONFIG['per_subject_class_quota']=subject_class_quota
        CONFIG['actual_recording_fractions']=fractions
        CONFIG['split_definition_sha256']=hashlib.sha256(json.dumps(
            sorted((sid,tid,sp) for (sid,tid),sp in split_for_recording.items()),
            separators=(',',':')).encode()).hexdigest()
        # No class labels, trial IDs, or subject IDs are supplied to WavLM.
        rows=[];bad=[];sha_seen={}
        for sid,tid,c,original_label in tqdm(parsed,total=len(parsed),desc='[AUDIT] PME4 WAVs',unit='trial',dynamic_ncols=True):
            split=split_for_recording[(sid,tid)]
            f=DATA_ROOT/f's{sid:02d}'/f's{sid:02d}'/f't{tid:03d}'/f's{sid:02d}_t{tid:03d}.wav'
            if not f.is_file():
                alt=DATA_ROOT/f's{sid:02d}'/f't{tid:03d}'/f's{sid:02d}_t{tid:03d}.wav'
                if alt.is_file():f=alt
                else:bad.append(str(f));continue
            info=sf.info(f)
            if info.frames<400 or info.samplerate<=0 or info.channels<1:
                raise ValueError(f'Invalid WAV header: {f} {info}')
            duration=float(info.frames/info.samplerate)
            if not .3<=duration<=60:raise ValueError(f'Unexpected WAV duration {duration:.2f}s: {f}')
            digest=sha256_file(f) if not args.skip_wav_hash else f'NO_HASH_{sid}_{tid}'
            if digest in sha_seen:
                other=sha_seen[digest]
                if other[0]!=split or other[1]!=c:
                    raise RuntimeError(f'Cross-split/cross-label identical WAV hash: {f} vs {other}')
            sha_seen[digest]=(split,c,str(f))
            rows.append({'recording_id':f's{sid:02d}_t{tid:03d}','subject_id':sid,'trial_id':tid,
                'emotion_id':c,'raw_emotion':original_label,'split':split,
                'relative_path':str(f.relative_to(DATA_ROOT)),'abs_path':str(f.resolve()),
                'orig_num_frames':info.frames,'orig_sr':info.samplerate,'orig_channels':info.channels,
                'orig_duration_s':duration,'sha256':digest})
        if bad:raise FileNotFoundError(f'{len(bad)} required WAVs missing: {bad[:12]}')
        df=pd.DataFrame(rows).sort_values(['subject_id','trial_id']).reset_index(drop=True)
        if df.recording_id.duplicated().any():raise RuntimeError('Repeated original recording IDs')
        if set(df.subject_id)!=set(range(1,12)):raise RuntimeError('Missing PME4 speakers')
        if df.groupby(['subject_id','trial_id']).split.nunique().max()!=1:
            raise RuntimeError('FATAL: same original recording crossed train/val/test')
        for sp in ('train','val','test'):
            group=df[df.split==sp]
            if set(group.subject_id)!=set(range(1,12)):
                raise RuntimeError(f'Not all 11 speakers represented in {sp}')
            per=group.groupby(['subject_id','emotion_id']).size().unstack(fill_value=0).reindex(columns=range(4),fill_value=0)
            if per.empty or (per<2).any().any():
                raise RuntimeError(f'Each speaker needs >=2 original trials per emotion in {sp}: {per}')
        df.to_csv(OUTPUT_ROOT/'audit'/'dataset_manifest.csv',index=False)
        df.groupby(['split','subject_id','emotion_id']).size().to_csv(OUTPUT_ROOT/'audit'/'split_subject_class_counts.csv')
        (OUTPUT_ROOT/'audit'/'subject_class_split_quotas.json').write_text(json.dumps(subject_class_quota,indent=2)+'\n')
        split_cols=['recording_id','subject_id','trial_id','emotion_id','relative_path','split','sha256']
        split_csv=df[split_cols].sort_values('recording_id').to_csv(index=False).encode('utf8')
        immutable=OUTPUT_ROOT/'immutable_split.csv'
        if immutable.exists() and immutable.read_bytes()!=split_csv:
            raise RuntimeError('Immutable trial split or WAV metadata changed. Use a fresh --output-dir')
        immutable.write_bytes(split_csv)
        digest=hashlib.sha256(split_csv).hexdigest()
        report={'protocol':CONFIG['protocol_id'],'split_sha256':digest,
            'recording_split_definition_sha256':CONFIG['split_definition_sha256'],
            'metadata_csv_sha256':sha256_file(csv_path),
            'counts':{sp:int((df.split==sp).sum()) for sp in ('train','val','test')},
            'class_counts':{sp:{LABEL_NAMES[c]:int(((df.split==sp)&(df.emotion_id==c)).sum())
                                 for c in range(4)} for sp in ('train','val','test')},
            'n_subjects_per_split':{sp:11 for sp in ('train','val','test')},
            'n_recordings_per_split':dict(counts_global),
            'per_subject_class_quotas':subject_class_quota,'actual_recording_fractions':fractions,
            'excluded_emotions':dict(skipped),'target_classes':LABEL_NAMES,
            'hash_audited':not args.skip_wav_hash,
            'speaker_overlap':True,'global_trial_id_disjoint':False,'original_recording_disjoint':True,
            'caveat':'Not a subject-independent benchmark. All 11 speakers occur in all partitions; '
                     'original recordings disjoint; trial-index and subject overlap are intentional.'}
        save_json(OUTPUT_ROOT/'audit'/'audit_summary.json',report)
        log('[AUDIT PASSED]',report['counts'],'recording fractions',fractions,
            'SHA',digest,'eleven speakers in EACH split; original recording overlap=0')
        return df,report

    manifest,audit_summary=audit_pme4()
    TRAIN_N=int((manifest.split=='train').sum())
    VAL_N=int((manifest.split=='val').sum())
    TEST_N=int((manifest.split=='test').sum())
    save_json(OUTPUT_ROOT/'resolved_config.json',CONFIG)
    if args.mode=='audit':
        log('[DONE] Audit only; no WavLM model loaded or trained.')
        return

    import transformers
    from transformers import WavLMModel,WavLMConfig,AutoFeatureExtractor
    from huggingface_hub import HfApi
    log('[MODEL] Python',platform.python_version(),'Torch',torch.__version__,
        'Transformers',transformers.__version__,
        'GPU',torch.cuda.get_device_name(0))
    if not re.fullmatch('[0-9a-f]{40}',CHECKPOINT_REVISION):raise ValueError('Need pinned pretrained checkpoint revision SHA')

    def stable_seed(*parts):
        b='|'.join(map(str,parts)).encode('utf8')
        return int.from_bytes(hashlib.sha256(b).digest()[:8],byteorder='little',signed=False)

    class AudioStore:
        def __init__(self,cache=True):
            self.cache=cache
            self.folder=OUTPUT_ROOT/'audio_cache';self.folder.mkdir(parents=True,exist_ok=True)
        def get(self,row):
            path=Path(row['abs_path']);cache=self.folder/(row['recording_id']+'_'+row['sha256'][:12]+'_16k.npy')
            if self.cache and cache.exists():
                arr=np.load(cache,mmap_mode='r',allow_pickle=False)
                if arr.ndim!=1 or len(arr)<400:raise RuntimeError('Corrupt cache: '+str(cache))
                return arr
            w,sr=sf.read(path,dtype='float32',always_2d=True)
            w=w.mean(axis=1,dtype=np.float32)
            if sr!=SAMPLE_RATE:
                g=math.gcd(int(sr),SAMPLE_RATE)
                w=resample_poly(w,SAMPLE_RATE//g,int(sr)//g,window=('kaiser',8.6)).astype(np.float32)
            if len(w)<400 or (not np.isfinite(w).all()) or float(np.max(np.abs(w)))<1e-8:
                raise ValueError('Bad waveform: '+str(path))
            if self.cache:
                tmp=cache.with_name(cache.name+f'.{os.getpid()}.tmp')
                with open(tmp,'wb') as f:np.save(f,w,allow_pickle=False)
                os.replace(tmp,cache)
                return np.load(cache,mmap_mode='r',allow_pickle=False)
            return w

    def expected_resampled_length(row):
        return math.ceil(int(row['orig_num_frames'])*SAMPLE_RATE/int(row['orig_sr']))

    def train_crop(row,wave,epoch):
        n=len(wave);maxstart=max(0,n-WINDOW_SAMPLES)
        region=(epoch+stable_seed(row['recording_id'])%5)%5
        anchor=round(maxstart*region/4)
        rng=np.random.default_rng(stable_seed(SEED,epoch,row['recording_id'],'train'))
        jitter=round(rng.uniform(-.35,.35)*SAMPLE_RATE)
        start=int(np.clip(anchor+jitter,0,maxstart))
        x=np.asarray(wave[start:start+WINDOW_SAMPLES],dtype=np.float32).copy()
        if rng.random()<.20:
            rms=np.sqrt(np.mean(x.astype(np.float64)**2))
            if rms>1e-5:
                noise_std=rms/(10**(rng.uniform(20.,35.)/20))
                x=np.clip(x+rng.normal(0,noise_std,len(x)).astype('float32'),-1.,1.)
        if len(x)<400:raise RuntimeError('Invalid train crop '+row['recording_id'])
        return x

    def deterministic_eval_windows(row):
        n=expected_resampled_length(row)
        if n<=WINDOW_SAMPLES:return [(0,n)]
        # 5-sec PME4 clips -> two overlapping 4-sec windows, both containing speech.
        n_windows=math.ceil(n/WINDOW_SAMPLES)
        maxstart=n-WINDOW_SAMPLES
        starts=np.linspace(0,maxstart,n_windows).round().astype(int)
        return [(int(x),int(x+WINDOW_SAMPLES)) for x in starts]

    class TrainCropDataset(Dataset):
        def __init__(self,df,store):
            self.rows=df.sort_values('recording_id').to_dict('records');self.store=store;self.epoch=0
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):
            row=self.rows[i];return {'wave':train_crop(row,self.store.get(row),self.epoch),
                'y':int(row['emotion_id']),'subject_id':int(row['subject_id']),
                'recording_id':row['recording_id']}

    class EvalWindowDataset(Dataset):
        def __init__(self,df,store):
            self.rows=df.sort_values('recording_id').to_dict('records');self.store=store
            self.items=[(i,start,stop) for i,row in enumerate(self.rows) for start,stop in deterministic_eval_windows(row)]
        def __len__(self):return len(self.items)
        def __getitem__(self,i):
            ridx,start,stop=self.items[i];row=self.rows[ridx];wave=self.store.get(row)
            if abs(len(wave)-expected_resampled_length(row))>1:
                raise ValueError('Resampled length mismatch: '+row['recording_id'])
            x=np.asarray(wave[start:min(stop,len(wave))],dtype=np.float32).copy()
            if len(x)<400:raise ValueError('Short eval crop '+row['recording_id'])
            return {'wave':x,'y':int(row['emotion_id']),'subject_id':int(row['subject_id']),
                    'recording_id':row['recording_id'],'trial_id':int(row['trial_id'])}

    def make_collate(extractor):
        def collate(items):
            features=extractor([x['wave'] for x in items],sampling_rate=SAMPLE_RATE,
                 return_tensors='pt',padding='max_length',max_length=WINDOW_SAMPLES,
                 truncation=True,return_attention_mask=True)
            if (features['attention_mask'].sum(dim=1)<400).any():
                raise ValueError('Invalid feature extraction mask')
            return {'input_values':features['input_values'],
                    'attention_mask':features['attention_mask'].long(),
                    'y':torch.tensor([x['y'] for x in items],dtype=torch.long),
                    'recording_id':[x['recording_id'] for x in items],
                    'subject_id':[x['subject_id'] for x in items],
                    'trial_id':[x.get('trial_id',-1) for x in items]}
        return collate

    class PME4BalancedSubjectPairSampler(Sampler):
        """Balanced per-class batches, positive same-class pairs from distinct speakers.

        PME4 may have uneven per-class recording counts across 11 training subjects.
        Shuffle without replacement within each speaker bucket; oversample only
        when required, with replacement after exhaustion. Every batch has two
        distinct speakers per target emotion. An epoch has ceil(max_class/2)
        pair rounds; all classes contribute equal numbers of examples per epoch.
        """
        def __init__(self,ds,batch_size=8,seed=42):
            self.ds=ds;self.batch_size=batch_size;self.seed=seed;self.epoch=0
            self.groups={c:defaultdict(list) for c in range(4)}
            for idx,row in enumerate(ds.rows):
                self.groups[int(row['emotion_id'])][int(row['subject_id'])].append(idx)
            for c in range(4):
                if len(self.groups[c])<2:raise ValueError('SupCon needs 2 train speakers per class')
            self.pair_rounds=math.ceil(max(sum(len(v) for v in self.groups[c].values()) for c in range(4))/2)
        def __len__(self):return self.pair_rounds*(1 if self.batch_size==8 else 2)
        def __iter__(self):
            rng=np.random.default_rng(stable_seed(self.seed,self.epoch,'pme4sampler'))
            streams={};pos={};counts=Counter()
            for c in range(4):
                for sid,idxs in self.groups[c].items():
                    key=(c,sid);streams[key]=list(idxs);rng.shuffle(streams[key]);pos[key]=0
            def next_index(c,sid):
                key=c,sid
                if pos[key]>=len(streams[key]):
                    rng.shuffle(streams[key]);pos[key]=0
                x=streams[key][pos[key]];pos[key]+=1
                return int(x)
            all_batches=[]
            for t in range(self.pair_rounds):
                pairs={}
                for c in range(4):
                    speakers=sorted(self.groups[c])
                    weights=np.asarray([len(self.groups[c][s])/(1+counts[(c,s)]) for s in speakers],dtype=float)
                    weights/=weights.sum()
                    ids=rng.choice(len(speakers),size=2,replace=False,p=weights)
                    ss=[speakers[int(i)] for i in ids]
                    counts[(c,ss[0])]+=1;counts[(c,ss[1])]+=1
                    pairs[c]=[next_index(c,ss[0]),next_index(c,ss[1])]
                if self.batch_size==8:
                    all_batches.append(sum([pairs[c] for c in range(4)],[]))
                else:
                    combos=[((0,1),(2,3)),((0,2),(1,3)),((0,3),(1,2))]
                    for c1,c2 in combos[(t+self.epoch)%3]:all_batches.append(pairs[c1]+pairs[c2])
            rng.shuffle(all_batches)
            for b in all_batches:
                rng.shuffle(b)
                if len(b)!=self.batch_size:raise AssertionError('Sampler bad batch size')
                for c in set(self.ds.rows[i]['emotion_id'] for i in b):
                    speakers=[self.ds.rows[i]['subject_id'] for i in b if self.ds.rows[i]['emotion_id']==c]
                    if len(speakers)!=2 or len(set(speakers))!=2:raise AssertionError('Bad positive speaker pair')
                yield b

    store=AudioStore(cache=CACHE_RESAMPLED_AUDIO)
    train_df=manifest.loc[manifest.split=='train'].copy()
    val_df=manifest.loc[manifest.split=='val'].copy()
    train_ds=TrainCropDataset(train_df,store)
    sampler=PME4BalancedSubjectPairSampler(train_ds,MICROBATCH_SIZE,SEED)
    TRAIN_SAMPLED_N=len(sampler)*MICROBATCH_SIZE
    TRAIN_UPDATE_COUNT=math.ceil(len(sampler)/GRAD_ACCUM)
    for epoch in (0,1,4,10,27,100):
        sampler.epoch=epoch; batches=list(iter(sampler))
        assert len(batches)==len(sampler)
        assert all(len(b)==MICROBATCH_SIZE for b in batches)
    log('[SAMPLER] Unique train recordings',TRAIN_N,'balanced samples/epoch',TRAIN_SAMPLED_N,
        'batch count',len(sampler),'optimizer steps/epoch',TRAIN_UPDATE_COUNT)
    class GroupedHierarchicalFusion(nn.Module):
        def __init__(self,hidden_size=1024):
            super().__init__()
            self.h=hidden_size
            self.layer_logits=nn.Parameter(torch.zeros(4,6))
            self.group_gate=nn.Linear(hidden_size,1)
            nn.init.zeros_(self.group_gate.weight);nn.init.zeros_(self.group_gate.bias)
            self.norm=nn.LayerNorm(hidden_size)
        def forward(self,hidden_states,mask):
            if len(hidden_states)!=24:raise RuntimeError(f'Expected 24 actual Transformer layers; got {len(hidden_states)}')
            if mask.dtype!=torch.bool:mask=mask.bool()
            valid=mask.float().sum(dim=1).clamp(min=1).unsqueeze(-1)
            if (~mask.any(dim=1)).any():raise ValueError('GHF received an all-pad audio sample')
            weights=F.softmax(self.layer_logits.float(),dim=-1)
            groups=[];gates=[]
            for gi in range(4):
                # This ordered group contains exactly Transformer layers 6g+1..6g+6.
                h=sum(hidden_states[gi*6+j]*weights[gi,j].to(hidden_states[0].dtype) for j in range(6))
                pooled=(h.float()*mask.unsqueeze(-1)).sum(dim=1)/valid
                gate=torch.sigmoid(self.group_gate(pooled))  # [B,1]
                groups.append(h);gates.append(gate)
            gate_tensor=torch.cat(gates,dim=-1)            # [B,4]
            denom=gate_tensor.sum(dim=-1,keepdim=True).clamp(min=1e-6)
            x=sum(groups[g]*gate_tensor[:,g,None,None].to(groups[g].dtype) for g in range(4))
            x=x/denom[:,None,:].to(x.dtype)
            return self.norm(x),weights,gate_tensor

    class DualPathTemporalPooling(nn.Module):
        def __init__(self,dim=1024,heads=8):
            super().__init__()
            self.query=nn.Parameter(torch.randn(1,1,dim)*0.02)
            self.attention=nn.MultiheadAttention(dim,heads,dropout=0.10,batch_first=True)
            self.stats_proj=nn.Linear(2*dim,dim)
            self.stats_norm=nn.LayerNorm(dim)
            self.bottleneck=nn.Linear(dim,512)
            self.output_norm=nn.LayerNorm(512)
        def forward(self,x,mask):
            b,t,h=x.shape
            if not mask.bool().any(dim=1).all():raise ValueError('DPTP received all-invalid frames')
            q=self.query.expand(b,-1,-1).to(x.dtype)
            a,_=self.attention(q,x,x,key_padding_mask=~mask.bool(),need_weights=False)
            a=a[:,0,:]
            valid=mask.to(torch.float32).unsqueeze(-1)
            count=valid.sum(dim=1).clamp(min=1)
            # Compute stats in FP32 for BF16 stability, then cast for learned projections.
            xf=x.float()
            mu=(xf*valid).sum(dim=1)/count
            var=(((xf-mu.unsqueeze(1))**2)*valid).sum(dim=1)/count
            std=(var.clamp(min=0)+1e-5).sqrt()
            s=F.silu(self.stats_norm(self.stats_proj(torch.cat([mu,std],dim=-1).to(x.dtype))))
            merged=a+s
            z=F.silu(self.output_norm(self.bottleneck(merged)))
            return z

    def conv_frame_lengths(input_len, kernels, strides):
        x=input_len.long()
        for k,s in zip(kernels,strides): x=torch.div(x-int(k),int(s),rounding_mode='floor')+1
        return x

    class ArchitectureA(nn.Module):
        def __init__(self,wavlm):
            super().__init__()
            self.wavlm=wavlm
            self.ghf=GroupedHierarchicalFusion(1024)
            self.dptp=DualPathTemporalPooling(1024,heads=8)
            self.emotion_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(0.30),nn.Linear(256,4))
            self.projector=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Linear(256,128))
            self.stage='frozen'
        def set_stage(self,stage):
            assert stage in ('frozen','finetune')
            self.stage=stage
            self.wavlm.requires_grad_(False)
            if stage=='finetune':
                for layer in self.wavlm.encoder.layers[18:24]:layer.requires_grad_(True)
            elif stage=='frozen':
                self.wavlm.eval()
        def train(self,mode=True):
            super().train(mode)
            if mode and self.stage=='frozen':self.wavlm.eval()
            return self
        def forward(self,input_values,attention_mask,need_projection=False):
            if self.stage=='frozen':
                with torch.no_grad():
                    output=self.wavlm(input_values,attention_mask=attention_mask,
                         output_hidden_states=True,return_dict=True)
            else:
                output=self.wavlm(input_values,attention_mask=attention_mask,
                         output_hidden_states=True,return_dict=True)
            hs=output.hidden_states
            if len(hs)!=25:raise RuntimeError(f'Incorrect hidden states count={len(hs)}; need embedding +24 layers')
            frame_len=hs[-1].shape[1]
            lengths=conv_frame_lengths(attention_mask.long().sum(dim=-1),
                self.wavlm.config.conv_kernel,self.wavlm.config.conv_stride)
            if torch.any(lengths<=0):raise ValueError('Too few actual audio samples to yield WavLM frames')
            if torch.any(lengths>frame_len):raise ValueError('Frame count exceeded WavLM feature length')
            frame_mask=torch.arange(frame_len,device=input_values.device)[None,:]<lengths[:,None]
            feat,layer_weights,group_gates=self.ghf(hs[1:25],frame_mask)
            z=self.dptp(feat,frame_mask)
            logits=self.emotion_head(z)
            projection=F.normalize(self.projector(z).float(),dim=-1,eps=1e-8) if need_projection else None
            return {'logits':logits,'embedding':z,'projection':projection,'layer_weights':layer_weights,'group_gates':group_gates}


    # Original architecture self-check before touching pretrained weights.
    with torch.no_grad():
        fusion=GroupedHierarchicalFusion(hidden_size=64)
        hs=tuple(torch.randn(2,10,64) for _ in range(24))
        valid=torch.tensor([[1]*10,[1]*7+[0]*3],dtype=torch.bool)
        out,weight,gates=fusion(hs,valid)
        assert out.shape==(2,10,64) and torch.allclose(weight.sum(1),torch.ones(4))
        assert DualPathTemporalPooling(64,heads=8)(out,valid).shape==(2,512)
    log('[MODEL] GHF/DPTP structural check passed')
    extractor=AutoFeatureExtractor.from_pretrained(CHECKPOINT_NAME,revision=CHECKPOINT_REVISION)
    assert extractor.sampling_rate==16000 and extractor.do_normalize
    conf=WavLMConfig.from_pretrained(CHECKPOINT_NAME,revision=CHECKPOINT_REVISION)
    assert conf.hidden_size==1024 and conf.num_hidden_layers==24
    conf.apply_spec_augment=False;conf.layerdrop=0.0
    wavlm=WavLMModel.from_pretrained(CHECKPOINT_NAME,revision=CHECKPOINT_REVISION,config=conf)
    model=ArchitectureA(wavlm).to(DEVICE)
    model.set_stage('frozen')
    try:model.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    except TypeError:model.wavlm.gradient_checkpointing_enable()
    collate_fn=make_collate(extractor)
    log('[MODEL] Loaded pinned WavLM-Large:',CHECKPOINT_REVISION,
        '| parameters',sum(x.numel() for x in model.parameters()),
        '| warmup trainable',sum(x.numel() for x in model.parameters() if x.requires_grad))
    CONFIG.update({'transformers_version':transformers.__version__,'torch':torch.__version__,
        'scipy':scipy.__version__,'model':'WavLM Large + exact EAV GHF/DPTP',
        'microbatch':MICROBATCH_SIZE,'grad_accum':GRAD_ACCUM,
        'train_recordings':TRAIN_N,'sampled_examples_per_epoch':TRAIN_SAMPLED_N,
        'validation_recordings':VAL_N,'test_recordings':TEST_N})
    save_json(OUTPUT_ROOT/'resolved_config.json',CONFIG)
    def supcon_loss(projections, targets, tau=0.07):
        z=F.normalize(projections.float(), dim=-1, eps=1e-8)
        y=targets.view(-1)
        sim=(z@z.T)/tau
        diag=torch.eye(len(y),dtype=torch.bool,device=y.device)
        positive=(y[:,None]==y[None,:]) & ~diag
        if not positive.any(dim=1).all():
            raise ValueError('SupCon batch has no positive for at least one anchor')
        sim=sim.masked_fill(diag, float('-inf'))
        logp=sim-torch.logsumexp(sim,dim=1,keepdim=True)
        loss=-torch.where(positive,logp,torch.zeros_like(logp)).sum(1)/positive.sum(1)
        if not torch.isfinite(loss).all():
            raise FloatingPointError('Non-finite SupCon loss')
        return loss.mean()

    def atomic_save_torch(path,payload):
        path=Path(path)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_name(path.name+'.tmp')
        torch.save(payload,temp)
        os.replace(temp,path)


    @torch.inference_mode()
    def infer_recordings(df,split,epoch=None,collect_embeddings=True,verbose=True):
        if split not in ('train','val','test') or set(df.split.unique())!={split}:
            raise ValueError('Mismatch between requested split and manifest')
        if split=='test' and not (args.mode=='test' and args.unlock_test):
            raise PermissionError('Test embargo is active')
        model.eval();ds=EvalWindowDataset(df,store)
        loader=DataLoader(ds,batch_size=EVAL_BATCH_SIZE,shuffle=False,num_workers=NUM_WORKERS,
                          pin_memory=True,collate_fn=collate_fn,persistent_workers=(NUM_WORKERS>0))
        sums={};emb_sum={};cnt=Counter()
        for b in tqdm(loader,desc=f'[EVAL] {split} full recording',unit='batch',dynamic_ncols=True,disable=not verbose):
            iv=b['input_values'].to(DEVICE,non_blocking=True)
            mask=b['attention_mask'].to(DEVICE,non_blocking=True)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                result=model(iv,mask,need_projection=False)
            logits=result['logits'].float().cpu().numpy()
            embeddings=result['embedding'].float().cpu().numpy() if collect_embeddings else None
            if not np.isfinite(logits).all() or (embeddings is not None and not np.isfinite(embeddings).all()):
                raise FloatingPointError('Nonfinite evaluation output')
            for j,rid in enumerate(b['recording_id']):
                if rid not in sums:
                    sums[rid]=np.zeros(4,dtype=np.float64)
                    if collect_embeddings:emb_sum[rid]=np.zeros(512,dtype=np.float64)
                sums[rid]+=logits[j]
                if collect_embeddings:emb_sum[rid]+=embeddings[j]
                cnt[rid]+=1
        if set(cnt)!={r['recording_id'] for r in ds.rows}:raise RuntimeError('Incomplete eval predictions')
        rows=[];arrays=[]
        for r in ds.rows:
            rid=r['recording_id']; n=cnt[rid]
            logit=sums[rid]/n;prob=softmax(logit);pred=int(np.argmax(logit))
            label=int(r['emotion_id'])
            rec={'recording_id':rid,'subject_id':int(r['subject_id']),'trial_id':int(r['trial_id']),
                'split':split,'true_class':label,'pred_class':pred,'correct':int(label==pred),
                'n_windows':n,'checkpoint_epoch':epoch}
            for i,nm in enumerate(LABEL_SLUGS):
                rec['p_'+nm]=float(prob[i]);rec['logit_'+nm]=float(logit[i])
            rows.append(rec)
            if collect_embeddings:arrays.append((emb_sum[rid]/n).astype('float32'))
        predictions=pd.DataFrame(rows).sort_values('recording_id').reset_index(drop=True)
        if len(predictions)!=len(df) or predictions.recording_id.duplicated().any():
            raise RuntimeError('Bad predictions shape or duplicate IDs')
        if collect_embeddings:
            arrmap={r['recording_id']:z for r,z in zip(rows,arrays)}
            embedding_array=np.stack([arrmap[x] for x in predictions.recording_id])
            assert embedding_array.shape==(len(predictions),512)
        else:embedding_array=None
        log(f'[EVAL] {split}: {len(predictions)} original recordings across {len(ds)} windows')
        return predictions,embedding_array

    def score_predictions(pred_df,split,bootstrap=True):
        expected=set(CONFIG['split'][split])
        if set(pred_df.subject_id)!=expected:raise ValueError(f'Subject IDs mismatch for {split}')
        if pred_df.recording_id.duplicated().any():raise ValueError('Duplicate eval recording')
        actual_ids=set(manifest.loc[manifest.split==split,'recording_id'])
        if set(pred_df.recording_id)!=actual_ids:raise ValueError('Missing/extra evaluation recording IDs')
        truth=pred_df.true_class.to_numpy(dtype=int);pred=pred_df.pred_class.to_numpy(dtype=int)
        if not np.array_equal(pred_df.correct.to_numpy(),(truth==pred).astype(int)):
            raise RuntimeError('Incorrect prediction scoreboard')
        subs=[];subs_class=[];sub_cm={}
        for sid,g in pred_df.groupby('subject_id',sort=True):
            y=g.true_class.to_numpy(int);yp=g.pred_class.to_numpy(int)
            p,r,f,s=precision_recall_fscore_support(y,yp,labels=range(4),zero_division=0)
            mat=confusion_matrix(y,yp,labels=range(4));sub_cm[int(sid)]=mat
            subs.append({'subject_id':int(sid),'n_recordings':len(g),
                'correct':int((y==yp).sum()),'accuracy':float((y==yp).mean()),
                'uar':float(r.mean()),'macro_f1':float(f.mean())})
            for c in range(4):
                subs_class.append({'subject_id':int(sid),'class_id':c,'emotion':LABEL_NAMES[c],
                    'precision':float(p[c]),'recall':float(r[c]),'f1':float(f[c]),'support':int(s[c])})
        sub_df=pd.DataFrame(subs);subclass_df=pd.DataFrame(subs_class)
        pp,rr,ff,ss=precision_recall_fscore_support(truth,pred,labels=range(4),zero_division=0)
        probs=pred_df[['p_neutral','p_sad','p_angry','p_happy']].to_numpy(dtype=float)
        cm=confusion_matrix(truth,pred,labels=range(4))
        out={'protocol':CONFIG['protocol_id'],'split':split,'primary_metric':'unweighted_mean_of_subject_accuracies',
             'mean_subject_accuracy':float(sub_df.accuracy.mean()),
             'pooled_recording_accuracy':float((truth==pred).mean()),
             'uar':float(rr.mean()),'macro_f1':float(ff.mean()),
             'weighted_f1':float(np.average(ff,weights=ss)),
             'mean_subject_macro_f1':float(sub_df.macro_f1.mean()),
             'mean_subject_uar':float(sub_df.uar.mean()),
             'subject_accuracy_std':float(sub_df.accuracy.std(ddof=1)),
             'n_subjects':len(sub_df),'n_recordings':len(pred_df),
             'n_correct':int((truth==pred).sum()),'n_incorrect':int((truth!=pred).sum()),
             'total_windows':int(pred_df.n_windows.sum()),
             'cross_entropy':float(log_loss(truth,probs,labels=list(range(4))))}
        try:out['auroc_macro_ovr']=float(roc_auc_score(truth,probs,multi_class='ovr',average='macro',labels=list(range(4))))
        except ValueError:out['auroc_macro_ovr']=None
        if bootstrap:
            rng=np.random.default_rng(20261106)
            x=sub_df.accuracy.to_numpy()
            boot=rng.choice(x,size=(10000,len(x)),replace=True).mean(1)
            out['subject_bootstrap_95ci']=[float(v) for v in np.percentile(boot,[2.5,97.5])]
        byclass=pd.DataFrame({'class_id':range(4),'emotion':LABEL_NAMES,
                              'precision':pp,'recall':rr,'f1':ff,'support':ss})
        return out,sub_df,subclass_df,byclass,cm,sub_cm
    def plot_heatmap(mat,labels_x,labels_y,title,dest,fmt='int',cmap='Blues',vmin=None,vmax=None):
        fig,ax=plt.subplots(figsize=(max(6,len(labels_x)*1.4),max(4,len(labels_y)*0.63+1.4)))
        im=ax.imshow(mat,aspect='auto',cmap=cmap,vmin=vmin,vmax=vmax)
        for i in range(len(labels_y)):
            for j in range(len(labels_x)):
                val=mat[i,j]
                shown=str(int(val)) if fmt=='int' else f'{float(val):.2f}'
                ax.text(j,i,shown,ha='center',va='center',fontsize=9,
                        color='white' if im.norm(val)>0.72 else 'black')
        ax.set_xticks(range(len(labels_x)),labels_x)
        ax.set_yticks(range(len(labels_y)),labels_y)
        ax.set_xlabel('Predicted emotion' if len(labels_x)==4 and len(labels_y)==4 else 'Emotion')
        ax.set_ylabel('Actual emotion' if len(labels_x)==4 and len(labels_y)==4 else 'Subject')
        ax.set_title(title)
        fig.colorbar(im,ax=ax,shrink=0.82)
        fig.tight_layout();fig.savefig(dest,dpi=170,bbox_inches='tight');plt.close(fig)

    def plot_evaluation(split, pred_df, stats, subjects, subject_classes, class_df, cm, cms, dest):
        dest.mkdir(parents=True,exist_ok=True)
        # SUBJECT ACCURACY BAR CHART WITH THE PRIMARY 7-SUBJECT MEAN
        fig,ax=plt.subplots(figsize=(10,5))
        xs=np.arange(len(subjects));vals=subjects.accuracy.to_numpy()*100
        bars=ax.bar(xs,vals,color='#377CA1',width=0.65)
        ax.axhline(stats['mean_subject_accuracy']*100,color='#AE3A35',ls='--',lw=2,
                   label=f"Mean of {len(subjects)} subjects = {stats['mean_subject_accuracy']*100:.2f}%")
        ax.set_xticks(xs,[f'S{int(s):02d}' for s in subjects.subject_id])
        ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)');ax.set_xlabel('Subject')
        ax.set_title(f'Architecture A — {split.upper()} — individual subject accuracy')
        ax.legend(loc='lower right');ax.grid(axis='y',alpha=.25)
        for rect,v in zip(bars,vals):ax.text(rect.get_x()+rect.get_width()/2,min(v+1,104),f'{v:.1f}%',ha='center',fontsize=9)
        fig.tight_layout();fig.savefig(dest/'subject_accuracy_bar.png',dpi=180);plt.close(fig)
        # SUBJECT ACCURACY, UAR AND MACRO F1
        fig,ax=plt.subplots(figsize=(11,5.3));width=.25
        for j,(field,label) in enumerate([('accuracy','Accuracy'),('uar','UAR'),('macro_f1','Macro F1')]):
            ax.bar(xs+(j-1)*width,subjects[field].to_numpy()*100,width,label=label)
        ax.set_xticks(xs,[f'S{int(s):02d}' for s in subjects.subject_id]);ax.set_ylim(0,105)
        ax.set_ylabel('Percent');ax.set_title(f'{split.upper()}: per-subject metric comparison')
        ax.legend();ax.grid(axis='y',alpha=.2)
        fig.tight_layout();fig.savefig(dest/'subject_accuracy_uar_f1.png',dpi=180);plt.close(fig)
        # CONFUSION MATRICES
        plot_heatmap(cm,LABEL_NAMES,LABEL_NAMES,f'{split.upper()} confusion matrix (all recordings)',dest/'confusion_counts.png')
        norm=cm/np.maximum(cm.sum(1,keepdims=True),1)
        plot_heatmap(norm,LABEL_NAMES,LABEL_NAMES,f'{split.upper()} row-normalized confusion',
                     dest/'confusion_normalized.png',fmt='float',vmin=0,vmax=1)
        # SUBJECT x CLASS HEATMAPS
        ids=list(subjects.subject_id.astype(int));nms=[f'S{sid:02d}' for sid in ids]
        for col in ['recall','f1','precision']:
            piv=subject_classes.pivot(index='subject_id',columns='class_id',values=col).loc[ids,[0,1,2,3]]
            plot_heatmap(piv.to_numpy(),LABEL_NAMES,nms,
                         f'{split.upper()} subject × emotion {col}',dest/f'subject_class_{col}_heatmap.png',
                         fmt='float',vmin=0,vmax=1)
        # CLASS-WISE PRECISION RECALL F1
        fig,ax=plt.subplots(figsize=(9,4.8));xx=np.arange(4);width=.25
        for j,(field,label) in enumerate([('precision','Precision'),('recall','Recall'),('f1','F1')]):
            ax.bar(xx+(j-1)*width,class_df[field].to_numpy()*100,width,label=label)
        ax.set_xticks(xx,LABEL_NAMES);ax.set_ylim(0,105)
        ax.set_ylabel('Percent');ax.set_title(f'{split.upper()} class-wise metrics')
        ax.grid(axis='y',alpha=.2);ax.legend()
        fig.tight_layout();fig.savefig(dest/'classwise_precision_recall_f1.png',dpi=180);plt.close(fig)
        # CONFIDENCE DISTRIBUTION (whether overconfident errors exist)
        probs=pred_df[['p_neutral','p_sad','p_angry','p_happy']].to_numpy()
        confid=probs.max(axis=1);correct=pred_df.correct.to_numpy(dtype=bool)
        fig,ax=plt.subplots(figsize=(8.5,4.4));edges=np.linspace(0,1,21)
        ax.hist(confid[correct],bins=edges,alpha=.7,label='Correct',color='#267A55')
        if (~correct).any():ax.hist(confid[~correct],bins=edges,alpha=.7,label='Incorrect',color='#C44848')
        ax.set_xlabel('Maximum predicted probability');ax.set_ylabel('Recordings')
        ax.set_title(f'{split.upper()} prediction confidence');ax.legend()
        fig.tight_layout();fig.savefig(dest/'prediction_confidence.png',dpi=180);plt.close(fig)
        # INDIVIDUAL SUBJECT CONFUSION MATRICES
        indiv=dest/'individual_subject_confusions';indiv.mkdir(exist_ok=True)
        for sid,sub_cm in sorted(cms.items()):
            plot_heatmap(sub_cm,LABEL_NAMES,LABEL_NAMES,f'Subject {sid} — {split.upper()} ({int(sub_cm.sum())} recordings)',
                         indiv/f'subject{sid:02d}_confusion.png')
        # ROC curves, using one-vs-rest probabilities
        from sklearn.metrics import roc_curve, auc
        fig,ax=plt.subplots(figsize=(6.2,5.4))
        for cls in range(4):
            truth=(pred_df.true_class.to_numpy()==cls).astype(int)
            fp,tp,_=roc_curve(truth,probs[:,cls]);a=auc(fp,tp)
            ax.plot(fp,tp,label=f'{LABEL_NAMES[cls]} (AUC {a:.3f})')
        ax.plot([0,1],[0,1],'--',color='gray',lw=1)
        ax.set_xlabel('False positive rate');ax.set_ylabel('True positive rate')
        ax.set_title(f'{split.upper()} one-vs-rest ROC');ax.legend(loc='lower right')
        fig.tight_layout();fig.savefig(dest/'roc_ovr.png',dpi=180);plt.close(fig)

    def export_evaluation(split,preds,embeddings,checkpoint_epoch):
        metrics,subjects,subject_class,by_class,cm,subject_cm=score_predictions(preds,split,bootstrap=True)
        folder=OUTPUT_ROOT/split
        folder.mkdir(parents=True,exist_ok=True)
        preds.to_csv(folder/'recording_predictions.csv',index=False)
        subjects.to_csv(folder/'per_subject_metrics.csv',index=False)
        subject_class.to_csv(folder/'per_subject_class_metrics.csv',index=False)
        by_class.to_csv(folder/'per_class_metrics.csv',index=False)
        pd.DataFrame(cm,index=LABEL_NAMES,columns=LABEL_NAMES).to_csv(folder/'confusion_counts.csv')
        pd.DataFrame(cm/np.maximum(cm.sum(1,keepdims=True),1),index=LABEL_NAMES,columns=LABEL_NAMES).to_csv(folder/'confusion_normalized.csv')
        if embeddings is not None:
            np.save(folder/'recording_embeddings_512d.npy',embeddings,allow_pickle=False)
            preds[['recording_id','subject_id','trial_id','true_class']].to_csv(folder/'embedding_recording_ids.csv',index=False)
        metrics['best_epoch']=int(checkpoint_epoch)
        metrics['split_sha256']=audit_summary['split_sha256']
        save_json(folder/'metrics.json',metrics)
        save_json(folder/'per_subject_metrics.json',subjects.to_dict(orient='records'))
        plot_evaluation(split,preds,metrics,subjects,subject_class,by_class,cm,subject_cm,folder)
        log(f'[EVAL {split.upper()}] HEADLINE MSA={metrics["mean_subject_accuracy"]:.4%} = '
            'mean of subject accuracies; '
            f'pooled={metrics["pooled_recording_accuracy"]:.4%} '
            f'UAR={metrics["uar"]:.4%} macroF1={metrics["macro_f1"]:.4%} '
            f'correct={metrics["n_correct"]}/{metrics["n_recordings"]}')
        for _,r in subjects.iterrows():
            log(f'[EVAL {split.upper()}] subject{int(r.subject_id):02d}: '
                f'{r.accuracy:.4%} ({int(r.correct)}/{int(r.n_recordings)}); '
                f'UAR {r.uar:.4%}; macro-F1 {r.macro_f1:.4%}')
        log(f'[EVAL {split.upper()}] Plots, predictions, metrics and embeddings saved to {folder}')
        return metrics

    def decay_eligible(name,p):
        return p.ndim>=2 and not name.endswith('bias') and 'layer_logits' not in name and 'group_gate' not in name

    def make_optimizer(stage):
        groups=defaultdict(list)
        for name,p in model.named_parameters():
            if not p.requires_grad:continue
            if name.startswith('wavlm.encoder.layers.'):
                idx=int(name.split('.')[3])
                if idx<18:raise RuntimeError('Frozen lower layer unexpectedly trainable: '+name)
                lr=5e-6*(0.8**(23-idx))
            else:
                lr=3e-4 if stage=='frozen' else 1e-4
            wd=0.01 if decay_eligible(name,p) else 0.
            groups[(lr,wd)].append(p)
        return torch.optim.AdamW([{'params':params,'lr':lr,'base_lr':lr,'weight_decay':wd}
                for (lr,wd),params in groups.items()],betas=(0.9,0.999),eps=1e-8)

    def make_scheduler(optimizer,stage):
        num_epochs=args.frozen_epochs if stage=='frozen' else args.max_epochs-args.frozen_epochs
        n_updates=TRAIN_UPDATE_COUNT*num_epochs
        warmup=max(1,int(round(n_updates*0.1)))
        def multiplier(s):
            if s<warmup:return max(0.05,(s+1)/warmup)
            progress=min(1.,(s-warmup)/max(1,n_updates-warmup))
            return 0.05+0.95*(1+math.cos(math.pi*progress))/2
        return torch.optim.lr_scheduler.LambdaLR(optimizer,multiplier)

    def model_state_cpu():
        return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}

    def capture_rng():
        return {'python':random.getstate(),'numpy':np.random.get_state(),
                'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all()}

    def restore_rng(d):
        random.setstate(d['python']);np.random.set_state(d['numpy'])
        torch.set_rng_state(d['torch']);torch.cuda.set_rng_state_all(d['cuda'])

    def checkpoint_common(epoch,stage,stats):
        return {'format':'PME4_ARCH_A_ALL11_TRIAL_70_15_15_V1','epoch':int(epoch),'stage':stage,
            'model':model_state_cpu(),'val_metrics':stats,
            'pretrained_revision':CHECKPOINT_REVISION,
            'split_sha256':audit_summary['split_sha256'],
            'split_definition_sha256':CONFIG['split_definition_sha256'],
            'protocol':CONFIG['protocol_id'],'run_config':{k:v for k,v in vars(args).items() if k not in ('resume','mode','unlock_test')}}

    def save_last(epoch,stage,optimizer,scheduler,stats,history,early):
        payload=checkpoint_common(epoch,stage,stats)
        payload.update({'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),
                        'rng':capture_rng(),'history':history,'early_stop':early})
        atomic_save_torch(OUTPUT_ROOT/'last.pth',payload)
        log('[CHECKPOINT] last.pth saved, epoch',epoch,'stage',stage)

    def save_best(epoch,stage,stats):
        payload=checkpoint_common(epoch,stage,stats)
        best_path=OUTPUT_ROOT/'best.pth'
        if best_path.exists():
            shutil.copy2(best_path,OUTPUT_ROOT/'best_previous.pth')
        atomic_save_torch(best_path,payload)
        log('[CHECKPOINT] best.pth saved, selected epoch',epoch,'MSA',f'{stats["mean_subject_accuracy"]:.4%}')

    def load_checkpoint(path,load_optimizer=False):
        data=torch.load(path,map_location='cpu',weights_only=False)  # local trusted checkpoint only
        if data.get('format')!='PME4_ARCH_A_ALL11_TRIAL_70_15_15_V1':
            raise RuntimeError('Incompatible checkpoint format; only PME4 all11 trial-disjoint V1 checkpoints may be loaded')
        if data['pretrained_revision']!=CHECKPOINT_REVISION:
            raise RuntimeError('Pretrained weight revision differs from checkpoint')
        if data['split_sha256']!=audit_summary['split_sha256']:
            raise RuntimeError('Dataset/split hash differs from checkpoint')
        if data['split_definition_sha256']!=CONFIG['split_definition_sha256']:
            raise RuntimeError('Subject-ID definition differs from checkpoint')
        if args.mode=='test' and data['run_config'].get('seed')!=args.seed:
            raise RuntimeError('Test invocation seed differs from original training config')
        if len(data['model'])!=len(model.state_dict()):
            raise RuntimeError('Missing parameters or incompatible model architecture')
        model.load_state_dict(data['model'],strict=True)
        model.set_stage(data['stage'])
        return data

    def log_training_status(epoch,steps,seen,loss_sum,correct,optimizer,start):
        lr_summary={f'{grp["base_lr"]:.3e}':f'{grp["lr"]:.3e}' for grp in optimizer.param_groups}
        log(f'[TRAIN PROGRESS] epoch={epoch}/{args.max_epochs} updates={steps}/{TRAIN_UPDATE_COUNT} '
            f'seen={seen}/{TRAIN_SAMPLED_N} loss={loss_sum/seen:.5f} accuracy={correct/seen:.3%} '
            f'lr_by_group={lr_summary} elapsed_s={time.perf_counter()-start:.1f}')

    def fit_epoch(epoch,optimizer,scheduler):
        model.train()
        train_ds.epoch=epoch-1;sampler.epoch=epoch-1
        loader=DataLoader(train_ds,batch_sampler=sampler,num_workers=NUM_WORKERS,pin_memory=True,
                          collate_fn=collate_fn,persistent_workers=(NUM_WORKERS>0))
        if len(loader)!=len(sampler):raise RuntimeError('Incorrect train loader size')
        if epoch<=args.frozen_epochs: sup_weight=0.
        else:sup_weight=min(.05,.05*(epoch-args.frozen_epochs)/3)
        total=ce_total=su_total=0.;correct=seen=0;steps=0;grad_sum=0.
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        start=time.perf_counter()
        bar=tqdm(loader,desc=f'[TRAIN] epoch {epoch}/{args.max_epochs}',unit='batch',dynamic_ncols=True,mininterval=1.0)
        for bi,batch in enumerate(bar):
            iv=batch['input_values'].to(DEVICE,non_blocking=True)
            mask=batch['attention_mask'].to(DEVICE,non_blocking=True)
            y=batch['y'].to(DEVICE,non_blocking=True)
            with torch.amp.autocast('cuda',dtype=torch.bfloat16):
                outputs=model(iv,mask,need_projection=bool(sup_weight))
            logits=outputs['logits'].float()
            ce=F.cross_entropy(logits,y,label_smoothing=.05)
            contrast=supcon_loss(outputs['projection'],y,tau=.07) if sup_weight else ce.new_zeros(())
            loss=ce+sup_weight*contrast
            if not torch.isfinite(loss):raise FloatingPointError(f'Nonfinite loss at epoch {epoch} batch {bi}')
            (loss/min(GRAD_ACCUM,len(loader)-(bi//GRAD_ACCUM)*GRAD_ACCUM)).backward()
            if (bi+1)%GRAD_ACCUM==0 or (bi+1)==len(loader):
                gn=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not torch.isfinite(gn):raise FloatingPointError('Nonfinite gradient norm')
                grad_sum+=float(gn)
                optimizer.step();scheduler.step();optimizer.zero_grad(set_to_none=True);steps+=1
                if steps%20==0 or steps==TRAIN_UPDATE_COUNT:
                    log_training_status(epoch,steps,seen+y.numel(),total+loss.item()*y.numel(),
                                        correct+int((logits.argmax(1)==y).sum()),optimizer,start)
            bs=y.numel();seen+=bs;correct+=int((logits.detach().argmax(1)==y).sum().item())
            total+=loss.item()*bs;ce_total+=ce.item()*bs;su_total+=contrast.item()*bs
            if bi%12==0:bar.set_postfix(loss=f'{total/seen:.4f}',acc=f'{correct/seen:.3f}',updates=steps)
        if seen!=TRAIN_SAMPLED_N or steps!=TRAIN_UPDATE_COUNT:
            raise RuntimeError(f'Train epoch incomplete: samples={seen},steps={steps}')
        return {'epoch':epoch,'stage':model.stage,'train_loss':total/seen,'train_ce':ce_total/seen,
                'train_supcon':su_total/seen,'train_crop_accuracy':correct/seen,
                'supcon_weight':sup_weight,'train_examples':seen,'optimizer_updates':steps,
                'mean_gradient_norm':grad_sum/max(steps,1),
                'seconds':time.perf_counter()-start,
                'peak_gpu_gib':torch.cuda.max_memory_allocated(DEVICE)/2**30,
                'last_head_lr':optimizer.param_groups[-1]['lr']}

    def plot_learning_curves(history):
        df=pd.DataFrame(history)
        if len(df)==0:return
        fig,ax=plt.subplots(figsize=(11,5))
        ax.plot(df.epoch,df.train_crop_accuracy*100,'o-',label='Training crop accuracy')
        ax.plot(df.epoch,df.val_msa*100,'s-',label='Full validation mean-subject accuracy')
        ax.plot(df.epoch,df.val_uar*100,'^-',label='Full validation UAR')
        ax.set_xlabel('Epoch');ax.set_ylabel('Percent');ax.set_ylim(0,105)
        ax.set_title('Architecture A full training — recording-level validation')
        ax.legend();ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(OUTPUT_ROOT/'training_accuracy_curves.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,5))
        ax.plot(df.epoch,df.train_loss,'o-',label='Training loss')
        ax.plot(df.epoch,df.val_cross_entropy,'s-',label='Full validation CE')
        ax.set_xlabel('Epoch');ax.set_ylabel('Loss')
        ax.set_title('Architecture A — training vs full-validation loss')
        ax.legend();ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(OUTPUT_ROOT/'training_loss_curves.png',dpi=180);plt.close(fig)
        fig,ax=plt.subplots(figsize=(11,4.8))
        ax.plot(df.epoch,df.mean_gradient_norm,'o-',label='Mean unclipped gradient norm')
        ax.set_xlabel('Epoch');ax.set_ylabel('Gradient norm');ax.set_title('Optimizer stability')
        ax.grid(alpha=.25)
        fig.tight_layout();fig.savefig(OUTPUT_ROOT/'gradient_norm_curve.png',dpi=180);plt.close(fig)

    def full_training():
        if (OUTPUT_ROOT/'training_complete.json').exists():
            raise RuntimeError('Training already completed here. Use a new output directory to retrain')
        if (OUTPUT_ROOT/'last.pth').exists() and not args.resume:
            raise RuntimeError('Existing last.pth found. Use --resume or a NEW --output-dir')
        if (OUTPUT_ROOT/'train_history.csv').exists() and not args.resume:
            raise RuntimeError('Existing training history. Use --resume or a NEW --output-dir')
        if args.resume and not (OUTPUT_ROOT/'last.pth').exists():
            raise FileNotFoundError('--resume needs existing last.pth in output dir')
        history=[];best_epoch=0;best_msa=-1.;not_improved=0;start_epoch=1
        best_rank=None;patience_reference=-1.
        opt=None;scheduler=None;last_stage=None;resume_data=None
        if args.resume:
            resume_data=load_checkpoint(OUTPUT_ROOT/'last.pth')
            prev=resume_data['run_config']
            compared=['batch_size','grad_accum','max_epochs','frozen_epochs','patience','min_delta','seed','no_cache']
            for k in compared:
                if prev.get(k)!=getattr(args,k):
                    raise RuntimeError(f'Resume config mismatch {k}: {prev.get(k)} vs {getattr(args,k)}')
            history=resume_data['history'];start_epoch=int(resume_data['epoch'])+1
            best_epoch=int(resume_data['early_stop']['best_epoch'])
            best_msa=float(resume_data['early_stop']['best_msa'])
            not_improved=int(resume_data['early_stop']['patience_count'])
            patience_reference=float(resume_data['early_stop'].get('patience_reference',best_msa))
            best_rank=tuple(resume_data['early_stop'].get('best_rank',(best_msa,0,0,0,-best_epoch)))
            # Crash recovery: if interrupted between best.pth and last.pth, the
            # latest best may belong to an uncommitted epoch; roll it back.
            current_best=OUTPUT_ROOT/'best.pth'
            if not current_best.exists():raise RuntimeError('Resuming requires best.pth')
            best_info=torch.load(current_best,map_location='cpu',weights_only=False)
            if int(best_info['epoch'])!=best_epoch:
                backup=OUTPUT_ROOT/'best_previous.pth'
                if not backup.exists():raise RuntimeError('Best checkpoint is ahead of last; no recovery copy')
                backup_info=torch.load(backup,map_location='cpu',weights_only=False)
                if int(backup_info['epoch'])!=best_epoch:
                    raise RuntimeError('Best/last checkpoint states inconsistent; start a new run')
                shutil.copy2(backup,current_best)
                log('[RESUME] Rolled back uncommitted best.pth to epoch',best_epoch)
            del best_info
            last_stage=resume_data['stage']
            restore_rng(resume_data['rng'])
            log(f'[RESUME] Continuing after completed epoch {start_epoch-1}; best epoch {best_epoch}, '
                f'best MSA {best_msa:.4%}; patience count {not_improved}')
        log('[TRAIN] Full training plan: epochs',args.max_epochs,'frozen epochs',args.frozen_epochs,
            'max fine-tune epochs',args.max_epochs-args.frozen_epochs,
            'patience',args.patience,'min_delta',args.min_delta)
        log('[TRAIN] All 11 speakers represented in VAL; TEST trial groups remain locked until explicit --mode test --unlock-test')
        for epoch in range(start_epoch,args.max_epochs+1):
            stage='frozen' if epoch<=args.frozen_epochs else 'finetune'
            if stage!=last_stage:
                model.set_stage(stage)
                opt=make_optimizer(stage);scheduler=make_scheduler(opt,stage)
                last_stage=stage
                if resume_data is not None:
                    if resume_data['stage']==stage:
                        opt.load_state_dict(resume_data['optimizer'])
                        scheduler.load_state_dict(resume_data['scheduler'])
                    # If stage changed (e.g. frozen epoch 4 -> finetune epoch 5),
                    # new optimizer/scheduler are expected; discard old state.
                    resume_data=None
                log('[TRAIN] Stage initialized:',stage,'trainable parameters:',
                    sum(p.numel() for p in model.parameters() if p.requires_grad))
            elif opt is None:
                model.set_stage(stage)
                opt=make_optimizer(stage);scheduler=make_scheduler(opt,stage)
                if resume_data is not None:
                    opt.load_state_dict(resume_data['optimizer'])
                    scheduler.load_state_dict(resume_data['scheduler'])
                    resume_data=None
            log('[TRAIN] Epoch',epoch,'/',args.max_epochs,'stage',stage,
                'train recordings',TRAIN_N,'full validation recordings',VAL_N)
            train_row=fit_epoch(epoch,opt,scheduler)
            val_preds,_=infer_recordings(val_df,'val',epoch=epoch,collect_embeddings=False)
            val_metrics,*_=score_predictions(val_preds,'val',bootstrap=False)
            row=dict(train_row)
            row.update({'val_msa':val_metrics['mean_subject_accuracy'],
                'val_uar':val_metrics['uar'],'val_macro_f1':val_metrics['macro_f1'],
                'val_cross_entropy':val_metrics['cross_entropy']})
            history.append(row)
            pd.DataFrame(history).to_csv(OUTPUT_ROOT/'train_history.csv',index=False)
            plot_learning_curves(history)
            msa=val_metrics['mean_subject_accuracy']
            # Best checkpoint: choose the HIGHEST actual MSA, not merely improvements
            # above early-stopping min_delta; tie-break by UAR, macro-F1, loss, earlier epoch.
            candidate_rank=(round(msa,12),round(val_metrics['uar'],12),
                            round(val_metrics['macro_f1'],12),-float(val_metrics['cross_entropy']),-epoch)
            if best_epoch==0 or candidate_rank>best_rank:
                best_rank=candidate_rank
                best_msa=msa;best_epoch=epoch
                save_best(epoch,stage,val_metrics)
                val_preds.to_csv(OUTPUT_ROOT/'best_validation_predictions.csv',index=False)
            if msa > patience_reference + args.min_delta:
                patience_reference=msa
                not_improved=0
            elif epoch>args.frozen_epochs:
                not_improved+=1
            early={'best_epoch':best_epoch,'best_msa':best_msa,'patience_count':not_improved,
                   'patience_reference':patience_reference,'best_rank':best_rank}
            save_last(epoch,stage,opt,scheduler,val_metrics,history,early)
            log(f'[EPOCH {epoch}] stage={stage} CE={train_row["train_ce"]:.5f} '
                f'train_crop_acc={train_row["train_crop_accuracy"]:.3%} '
                f'FULL VAL MSA={msa:.3%} UAR={val_metrics["uar"]:.3%} '
                f'macro-F1={val_metrics["macro_f1"]:.3%} '
                f'best_epoch={best_epoch} best_MSA={best_msa:.3%} '
                f'patience={not_improved}/{args.patience} '
                f'GPUpeak={train_row["peak_gpu_gib"]:.2f}GiB '
                f'epoch_seconds={train_row["seconds"]:.1f}')
            if epoch>args.frozen_epochs and not_improved>=args.patience:
                log('[EARLY STOP] Full validation MSA did not improve sufficiently for',args.patience,'epochs')
                break
        if not (OUTPUT_ROOT/'best.pth').is_file():raise RuntimeError('No best.pth after training')
        best=load_checkpoint(OUTPUT_ROOT/'best.pth')
        log('[TRAIN] Evaluating BEST model across all original PME4 validation recordings for final export')
        val_preds,val_emb=infer_recordings(val_df,'val',epoch=best_epoch,collect_embeddings=True)
        best_metrics=export_evaluation('val',val_preds,val_emb,best_epoch)
        # An optional, strictly in-sample training embedding export. Never fit
        # another probability-stacking model using these as OOF predictions.
        if args.export_train_embeddings:
            log('[EMBED] Exporting train embeddings from best checkpoint (IN SAMPLE, not OOF)')
            train_preds,train_emb=infer_recordings(train_df,'train',epoch=best_epoch,collect_embeddings=True)
            np.save(OUTPUT_ROOT/'train_recording_embeddings_512d_IN_SAMPLE.npy',train_emb,allow_pickle=False)
            train_preds.to_csv(OUTPUT_ROOT/'train_recording_embedding_ids_IN_SAMPLE.csv',index=False)
        with torch.no_grad():
            w=F.softmax(model.ghf.layer_logits.float(),dim=-1).cpu().numpy()
        pd.DataFrame([{'group':gi+1,'layer':gi*6+j+1,'weight':float(w[gi,j])}
            for gi in range(4) for j in range(6)]).to_csv(OUTPUT_ROOT/'ghf_layer_weights.csv',index=False)
        finish={'completed':True,'max_epochs':args.max_epochs,'epochs_actually_run':len(history),
                'best_epoch':best_epoch,'best_val_metrics':best_metrics,
                'best_checkpoint':str(OUTPUT_ROOT/'best.pth'),
                'last_checkpoint':str(OUTPUT_ROOT/'last.pth'),
                'test_embargoed':True,
                'split_sha256':audit_summary['split_sha256'],
                'caveat':'Speaker-overlapping trial-disjoint PME4 evaluation, posed same-sentence speech; not subject-independent or spontaneous.'}
        save_json(OUTPUT_ROOT/'training_complete.json',finish)
        log('[DONE] Training/validation complete. Final test remains LOCKED. '
            'Now, and only now, run --mode test --unlock-test for the TEST recordings of all 11 PME4 speakers.')

    def final_test():
        if not args.unlock_test:raise PermissionError('Final testing requires explicit --unlock-test')
        train_finished=OUTPUT_ROOT/'training_complete.json'
        if not train_finished.exists():raise RuntimeError('Training not completed; final test is unavailable')
        lock=OUTPUT_ROOT/'test'/'final_test_completed.json'
        if lock.exists():raise RuntimeError('Final test has already been run; do not tune/repeat on locked test')
        info=json.loads(train_finished.read_text())
        if info['split_sha256']!=audit_summary['split_sha256']:
            raise RuntimeError('Training/test split hash mismatch')
        data=load_checkpoint(OUTPUT_ROOT/'best.pth')
        epoch=int(data['epoch'])
        log('[FINAL TEST] Best checkpoint loaded at epoch',epoch)
        test_df=manifest[manifest.split=='test'].copy()
        preds,embeddings=infer_recordings(test_df,'test',epoch=epoch,collect_embeddings=True)
        metrics=export_evaluation('test',preds,embeddings,epoch)
        receipt={'test_evaluated_once':True,'date_utc':__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
                 'checkpoint':str(OUTPUT_ROOT/'best.pth'), 'checkpoint_sha256':sha256_file(OUTPUT_ROOT/'best.pth'),
                 'test_metric':'mean_of_eleven_individual_subject_accuracies',
                 'msa':metrics['mean_subject_accuracy'],
                 'subject_ids':list(range(1,12)),
                 'split_sha256':audit_summary['split_sha256'],
                 'warning':'Test set must not be used for subsequent architecture/parameter selection.'}
        save_json(lock,receipt)
        log('[FINAL TEST DONE] Mean subject accuracy =',f'{metrics["mean_subject_accuracy"]:.4%}',
            '| individual subject metrics + figures in',OUTPUT_ROOT/'test')

    if args.mode=='train':full_training()
    elif args.mode=='test':final_test()
    else:raise RuntimeError('Unhandled mode '+args.mode)

if __name__=='__main__':
    ARGS=parse_args()
    try:
        install_missing(ARGS)
        main()
    except KeyboardInterrupt:
        print('[INTERRUPTED] Checkpoints written at completed epochs remain valid.',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc()
        folder=ARGS.output_dir or ARGS.project_root/'outputs'/'pme4_architecture_a_all11_trial_70_15_15_v2_fixed'
        try:
            folder.mkdir(parents=True,exist_ok=True)
            with (folder/'run_failure.txt').open('a') as f:f.write(traceback.format_exc()+'\n')
        except Exception:pass
        sys.exit(1)
