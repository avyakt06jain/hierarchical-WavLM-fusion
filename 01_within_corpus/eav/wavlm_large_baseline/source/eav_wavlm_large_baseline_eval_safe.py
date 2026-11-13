#!/usr/bin/env python3
"""EAV full five-class WavLM-Large simple baseline, 70/15/15 recordings.

Exact original EAV split: 42 speakers x 5 emotions x (14 train,3 val,3 test).
Original speaking trials, NOT independent 5s windows. Speaker overlap is
intentional and must not be described as LOSO or speaker independent.
Label codes are verified against EAV EEG metadata as in eav_common.py.
Usage: --mode self-test | audit | train [--resume] | test --unlock-test | report
"""
from __future__ import annotations
import argparse, contextlib, datetime, hashlib, json, math, os, random, re, sys, time
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io as sio
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import softmax
from sklearn.metrics import (confusion_matrix,precision_recall_fscore_support,roc_auc_score,log_loss)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset,DataLoader,Sampler
from tqdm.auto import tqdm

DATASET='eav'
DEFAULT_DATA_ROOT=Path('/workspace/data/EAV')
OUTPUT_BASE=Path('/workspace/avyakt-sc-final/sc-final-speech-model/outputs/eav_wavlm_large_baseline')
LABELS=('Neutral','Sadness','Anger','Happiness','Calmness')
SLUGS=tuple(x.lower() for x in LABELS)
PROTOCOL='EAV_42_SUBJECT_OVERLAPPING_ORIGINAL_SPEAKING_RECORDING_DISJOINT_STRATIFIED_70_15_15_SEED42'
DEFAULT_WINDOW_SECONDS=5.
DEFAULT_FROZEN_BATCH=10
DEFAULT_FINETUNE_BATCH=4
DEFAULT_EVAL_BATCH=8
DEFAULT_EFFECTIVE_BATCH=20
DEFAULT_EPOCHS=30
DEFAULT_FROZEN_EPOCHS=4
DEFAULT_PATIENCE=8
DEFAULT_NOISE=.10
DEFAULT_LABEL_SMOOTHING=.03


def eav_audio_filename(name):
    # The original published files: 002_Trial_02_Speaking_Neutral_Aud.wav
    m=re.fullmatch(r'(\d+)_Trial_(\d+)_(?:cond_)?Speaking_(Neutral|Sadness|Anger|Happiness|Calmness)_Aud\.wav',
                   name,re.IGNORECASE)
    if m is None:
        raise ValueError('Unexpected EAV speaking filename: '+name)
    label={s.lower():i for i,s in enumerate(LABELS)}[m.group(3).lower()]
    return int(m.group(1)),int(m.group(2)),label


def eav_onehot_labels(path):
    mat=sio.loadmat(path)
    if 'label' not in mat:
        raise ValueError(f'MAT {path} lacks required one-hot label key')
    data=np.asarray(mat['label'])
    if data.shape==(200,10):data=data.T
    if data.shape!=(10,200):
        raise ValueError(f'MAT {path} label shape {data.shape}, expected 10x200')
    if not np.allclose(data.sum(0),1) or not np.all((data==0)|(data==1)):
        raise ValueError(f'Invalid EAV EEG one-hot label coding: {path}')
    return np.argmax(data,axis=0).astype(int)


def build_dataset_manifest(a, labels):
    root=a.data_root
    subjects=sorted([p for p in root.glob('subject*') if p.is_dir() and re.fullmatch(r'subject\d+',p.name)],
                    key=lambda p:int(p.name.replace('subject','')))
    sids={int(p.name.replace('subject','')) for p in subjects}
    if sids!=set(range(1,43)):
        raise ValueError(f'EAV subject mismatch missing={sorted(set(range(1,43))-sids)} extra={sorted(sids-set(range(1,43)))}')
    rows=[];sources=[]
    for sub in tqdm(subjects,desc='[AUDIT EAV]',unit='subject'):
        sid=int(sub.name.replace('subject',''))
        aud=sub/'Audio';eeg=sub/'EEG'
        if not aud.is_dir() or not eeg.is_dir():raise FileNotFoundError(f'EAV Audio/EEG missing: {sub}')
        files=sorted(p for p in aud.iterdir() if p.suffix.lower()=='.wav')
        raw_mats=sorted(p for p in eeg.glob('*.mat') if not p.stem.lower().endswith('_label'))
        label_mats=sorted(p for p in eeg.glob('*.mat') if p.stem.lower().endswith('_label'))
        if len(files)!=100 or len(raw_mats)!=1 or len(label_mats)!=1:
            raise ValueError(f'EAV speaker {sid}: expected 100 WAV, 1 EEG MAT, 1 EEG label MAT; got {len(files)},{len(raw_mats)},{len(label_mats)}')
        code=eav_onehot_labels(label_mats[0]);sources.append(label_mats[0])
        eeg_shape=[(n,s) for n,s,_ in sio.whosmat(raw_mats[0]) if n in ('seg','seg1')]
        if len(eeg_shape)!=1 or sorted(eeg_shape[0][1])!=[30,200,10000]:
            raise ValueError(f'EAV EEG shape/variable invalid: {raw_mats[0]} {eeg_shape}')
        seen=set();count=Counter();class_codes={};label_codes={}
        for f in files:
            trial,conversation,emotion=eav_audio_filename(f.name)
            if not 1<=trial<=200 or trial in seen:
                raise ValueError(f'EAV duplicated/invalid speaking trial: {f}')
            seen.add(trial);count[emotion]+=1
            eeg_label=int(code[trial-1])
            if eeg_label in class_codes and class_codes[eeg_label]!=emotion:
                raise ValueError(f'EAV label {eeg_label} maps to conflicting emotions for {sub}')
            if emotion in label_codes and label_codes[emotion]!=eeg_label:
                raise ValueError(f'EAV emotion {emotion} maps to multiple EEG labels for {sub}')
            class_codes[eeg_label]=emotion;label_codes[emotion]=eeg_label
            info=sf.info(f);duration=info.frames/info.samplerate
            if info.frames<1600 or not 5<=duration<=30:
                raise ValueError(f'Invalid EAV speech duration: {f} {duration}')
            st=f.stat()
            rows.append({'recording_id':f's{sid:02d}_i{trial:03d}',
                         'subject_id':sid,'trial_id':trial,'conversation_trial':conversation,
                         'emotion_id':emotion,'emotion':LABELS[emotion],
                         'eeg_label_code':eeg_label,'split':'PENDING',
                         'audio_path':str(f.resolve()),'audio_bytes':st.st_size,
                         'audio_mtime_ns':st.st_mtime_ns,'orig_sr':info.samplerate,
                         'orig_frames':info.frames,'duration_seconds':duration,
                         'sha256':sha_file(f)})
        if set(count)!=set(range(5)) or set(count.values())!={20} or len(class_codes)!=5:
            raise ValueError(f'EAV speaker {sid} expected 20 per class and 5 unique label codes; got {count}')
    df=pd.DataFrame(rows)
    if len(df)!=4200:
        raise RuntimeError(f'EAV expected 4200 original recordings; got {len(df)}')
    partition={}
    for (sid,c),subset in df.groupby(['subject_id','emotion_id'],sort=True):
        keys=sorted(subset.recording_id.tolist())
        rng=np.random.default_rng(42+int(sid)*10007+int(c)*997)
        ix=rng.permutation(len(keys))
        for part,positions in [('train',ix[:14]),('val',ix[14:17]),('test',ix[17:])]:
            for j in positions:partition[keys[j]]=part
    df['split']=df.recording_id.map(partition)
    if df.split.isna().any():raise RuntimeError('Unassigned EAV original recording')
    return df,sources


# === Shared baseline implementation (embedded in each stand-alone script) ===

def sha_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fd:
        for block in iter(lambda: fd.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def json_save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')
    os.replace(temp, path)


def seed_from(*pieces):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,pieces)).encode()).digest()[:8], 'little')


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fingerprint(frame):
    keys = ['recording_id', 'subject_id', 'trial_id', 'emotion_id', 'split', 'sha256']
    for name in ('eeg_label_code',):
        if name in frame.columns:
            keys.append(name)
    return hashlib.sha256(frame.sort_values('recording_id')[keys].to_csv(index=False).encode()).hexdigest()


def validate_manifest(frame, labels, report_dir):
    """Do all hash/identity checks before saving an immutable split."""
    report_dir.mkdir(parents=True, exist_ok=True)
    duplicated = frame[frame.duplicated('sha256', keep=False)].sort_values('sha256')
    duplicated.to_csv(report_dir / 'duplicate_hash_groups.csv', index=False)
    problems = []
    if frame.recording_id.duplicated().any():
        problems.append('Duplicated recording IDs')
    if frame.split.isna().any():
        problems.append('Unassigned split values')
    if set(frame.split.unique()) != {'train', 'val', 'test'}:
        problems.append('Missing train/val/test recordings')
    benign_duplicates=[]
    if duplicated.shape[0]:
        for _, group in duplicated.groupby('sha256'):
            if (group.split.nunique()==1 and group.subject_id.nunique()==1
                    and group.emotion_id.nunique()==1):
                benign_duplicates.append(group.recording_id.tolist())
            else:
                problems.append(f"Cross-split, cross-speaker or conflicting-label identical WAVs: {group.recording_id.tolist()[:8]}")
    if benign_duplicates:
        print(f'[AUDIT CAUTION] {len(benign_duplicates)} same-speaker/class/split identical-hash groups retained and documented',flush=True)
    for part in ('train', 'val', 'test'):
        piece = frame[frame.split == part]
        missing = set(range(len(labels))) - set(piece.emotion_id)
        if missing:
            problems.append(f'{part} missing classes {sorted(missing)}')
    json_save(report_dir / 'manifest_validation.json', {
        'valid': not bool(problems), 'problems': problems,
        'recordings': int(len(frame)), 'speakers': int(frame.subject_id.nunique()),
        'split_counts': {s:int((frame.split == s).sum()) for s in ('train','val','test')},
        'split_speaker_counts': {s:int(frame.loc[frame.split == s,'subject_id'].nunique()) for s in ('train','val','test')},
        'labels': list(labels), 'subject_overlap_by_design': True,
        'original_recording_disjoint': not frame.recording_id.duplicated().any(),
        'byte_hash_duplicates': int(len(duplicated)),
        'benign_same_subject_emotion_split_duplicate_groups':benign_duplicates,
    })
    frame.groupby(['split','emotion_id']).size().unstack(fill_value=0).to_csv(report_dir / 'split_by_emotion.csv')
    frame.groupby(['split','subject_id','emotion_id']).size().to_csv(report_dir / 'split_by_subject_emotion.csv')
    if problems:
        raise RuntimeError('Dataset integrity failed: ' + '; '.join(problems[:8]) +
            f'. Full report: {report_dir / "manifest_validation.json"}')


def load_or_audit(a, out, labels):
    report_dir = out / 'audit'
    a._report_dir = report_dir
    manifest_file = report_dir / 'immutable_manifest.csv'
    if a.mode != 'audit' and manifest_file.exists() and not a.refresh_audit:
        df = pd.read_csv(manifest_file)
        # Fast verification of source files; heavy SHA256 was done during first audit.
        missing = []
        changed = []
        for r in df.itertuples(index=False):
            file = Path(r.audio_path)
            if not file.is_file():
                missing.append(str(file)); continue
            st = file.stat()
            if st.st_size != int(r.audio_bytes) or st.st_mtime_ns != int(r.audio_mtime_ns):
                changed.append(str(file))
        if missing or changed:
            raise RuntimeError(f'Source audio modified/missing. Missing={missing[:3]} changed={changed[:3]}; run --mode audit --refresh-audit in a NEW output dir')
        # Metadata is scanned from original sources only during audit; preserve source fingerprints.
        meta = json.loads((report_dir / 'source_metadata.json').read_text())
        for original, saved in meta.get('source_paths_sha256', {}).items():
            if sha_file(original) != saved:
                raise RuntimeError(f'Dataset metadata changed since original audit: {original}')
        validate_manifest(df, labels, report_dir)
        print(f'[AUDIT REUSED] {len(df)} immutable recordings; split SHA={fingerprint(df)}', flush=True)
        return df
    frame, source_paths = build_dataset_manifest(a, labels)
    frame = frame.sort_values('recording_id').reset_index(drop=True)
    validate_manifest(frame, labels, report_dir)
    if manifest_file.exists():
        existing = pd.read_csv(manifest_file)
        cols = ['recording_id','subject_id','trial_id','emotion_id','split','sha256']
        if existing[cols].to_csv(index=False) != frame[cols].to_csv(index=False):
            raise RuntimeError('Immutable manifest changed. Choose a fresh --output-dir, do not overwrite the previous run.')
    else:
        frame.to_csv(manifest_file, index=False)
    json_save(report_dir / 'source_metadata.json', {
        'data_root': str(a.data_root.resolve()),
        'source_paths_sha256': {str(p): sha_file(p) for p in source_paths},
        'recording_split_sha256': fingerprint(frame),
        'model': 'WavLM-Large + masked mean + LayerNorm + Dropout(0.30) + Linear(N)',
        'research_caveat': 'Speakers overlap between partitions; test is recording-disjoint but not speaker-independent.'
    })
    print(f'[AUDIT PASSED] records={len(frame)} splits={frame.split.value_counts().to_dict()} sha={fingerprint(frame)}', flush=True)
    return frame


class AudioCache:
    def __init__(self, folder, enabled=True):
        self.folder = Path(folder)
        self.folder.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def get(self, row):
        target = self.folder / f"{row['recording_id']}_{row['audio_bytes']}_{row['audio_mtime_ns']}_16k.npy"
        if self.enabled and target.is_file():
            w = np.load(target, mmap_mode='r', allow_pickle=False)
            if w.ndim != 1 or len(w) < 1600:
                raise ValueError(f'Corrupt resampling cache: {target}')
            return w
        x, rate = sf.read(row['audio_path'], dtype='float32', always_2d=True)
        x = x.mean(axis=1, dtype=np.float32)
        if rate != 16000:
            factor = math.gcd(int(rate), 16000)
            x = resample_poly(x, 16000 // factor, rate // factor, window=('kaiser',8.6)).astype(np.float32)
        if len(x) < 1600 or not np.isfinite(x).all() or float(np.max(np.abs(x))) < 1e-8:
            raise ValueError('Invalid audio waveform: ' + row['audio_path'])
        if self.enabled:
            temp = target.with_name(target.name + f'.{os.getpid()}.tmp')
            with open(temp, 'wb') as fd:
                np.save(fd, x, allow_pickle=False)
            os.replace(temp, target)
            return np.load(target, mmap_mode='r', allow_pickle=False)
        return x


def all_eval_starts(n, width):
    if n <= width:
        return [0]
    # Exactly the original EAV/PME4 Architecture A multi-view recording coverage.
    num = math.ceil(n / width)
    return np.linspace(0, n - width, num).round().astype(int).tolist()


class TrainingAudio(Dataset):
    def __init__(self, df, store, width, seed, noise_probability):
        self.rows = df.sort_values('recording_id').to_dict('records')
        self.store = store
        self.width = width
        self.seed = seed
        self.noise = noise_probability
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        w = self.store.get(r)
        rng = np.random.default_rng(seed_from(self.seed,self.epoch,r['recording_id'],'train'))
        if DATASET == 'pme4':
            # Exact 5-region PME4 Architecture A training window rotation.
            max_start=max(0,len(w)-self.width)
            region=(self.epoch+seed_from(r['recording_id'])%5)%5
            anchor=round(max_start*region/4)
            start=int(np.clip(anchor+round(rng.uniform(-.35,.35)*16000),0,max_start))
        else:
            # Exact EAV original-recording epoch rotation with 0.2-s jitter.
            starts = all_eval_starts(len(w), self.width)
            region = (self.epoch + seed_from(r['recording_id'],'region') % len(starts)) % len(starts)
            start = int(np.clip(starts[region] + round(rng.uniform(-.20,.20)*16000),
                                0, max(0,len(w)-self.width)))
        x = np.asarray(w[start:start+self.width],dtype=np.float32).copy()
        if self.noise > 0 and rng.random() < self.noise:
            rms = float(np.sqrt(np.mean(x.astype(np.float64)**2)))
            if rms > 1e-6:
                snr_range=(20.,35.) if DATASET=='pme4' else (25.,40.)
                noise = rng.normal(0,rms/10**(rng.uniform(*snr_range)/20),len(x)).astype(np.float32)
                x = np.clip(x + noise,-1,1)
        return {'wave':x, 'y':int(r['emotion_id']), 'recording_id':r['recording_id']}


class EvaluationAudio(Dataset):
    def __init__(self, df, store, width):
        self.rows = df.sort_values('recording_id').to_dict('records')
        self.store = store
        self.width = width
        self.items = []
        for i,row in enumerate(self.rows):
            length = len(store.get(row))
            self.items.extend((i,k) for k in all_eval_starts(length,width))

    def __len__(self):
        return len(self.items)

    def __getitem__(self,i):
        rid,start = self.items[i]
        r = self.rows[rid]
        w = self.store.get(r)
        return {'wave': np.asarray(w[start:start+self.width],dtype=np.float32).copy(),
                'y':int(r['emotion_id']), 'recording_id':r['recording_id']}


class BalancedSpeakerPairs(Sampler):
    """Class-balanced sampled batches, each positive pair from two original speakers.

    This preserves the source class/speaker sampling intent of Architecture A,
    although this baseline is trained with cross-entropy only (no SupCon).
    """
    def __init__(self, ds, batch_size, n_batches, seed):
        if batch_size < 2 or batch_size % 2:
            raise ValueError('Batch size must be even for class/speaker-balanced pair sampler')
        self.ds, self.batch_size, self.n_batches, self.seed = ds, batch_size, n_batches, seed
        self.epoch = 0
        self.classes = sorted({int(r['emotion_id']) for r in ds.rows})
        self.groups = {c:defaultdict(list) for c in self.classes}
        for idx,r in enumerate(ds.rows):
            self.groups[int(r['emotion_id'])][int(r['subject_id'])].append(idx)
        if any(len(self.groups[c]) < 2 for c in self.classes):
            raise RuntimeError('Need at least two speakers in each emotion for balanced sampling')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(seed_from(self.seed,self.epoch,'balanced_pairs'))
        queues = {}; pos = {}; count = Counter()
        for c in self.classes:
            for sid,idxs in self.groups[c].items():
                key = c,sid
                queues[key] = idxs.copy(); rng.shuffle(queues[key]); pos[key] = 0
        schedule = self.classes * (math.ceil(self.n_batches*(self.batch_size//2)/len(self.classes))+1)
        schedule = schedule[:self.n_batches*(self.batch_size//2)]
        # Shuffle the order of class-pair cycles, not the class frequencies.
        cycles = [schedule[j:j+len(self.classes)] for j in range(0,len(schedule),len(self.classes))]
        rng.shuffle(cycles)
        scheduled_classes = [c for cycle in cycles for c in cycle]
        for i in range(self.n_batches):
            indices = []
            for c in scheduled_classes[i*(self.batch_size//2):(i+1)*(self.batch_size//2)]:
                speakers = sorted(self.groups[c])
                weights = np.asarray([len(self.groups[c][s]) / (1 + count[(c,s)]) for s in speakers],dtype=float)
                weights = weights / weights.sum()
                selected = rng.choice(speakers,size=2,replace=False,p=weights)
                for sid in selected:
                    sid = int(sid); key = c,sid
                    if pos[key] == len(queues[key]):
                        rng.shuffle(queues[key]); pos[key] = 0
                    indices.append(int(queues[key][pos[key]])); pos[key] += 1
                    count[key] += 1
            rng.shuffle(indices)
            yield indices


def make_collate(extractor, width):
    def collate(items):
        feats = extractor([x['wave'] for x in items],sampling_rate=16000,
                          return_tensors='pt',padding='max_length',max_length=width,
                          truncation=True,return_attention_mask=True)
        return {'input_values':feats['input_values'],
                'attention_mask':feats['attention_mask'].long(),
                'y':torch.tensor([x['y'] for x in items],dtype=torch.long),
                'recording_id':[x['recording_id'] for x in items]}
    return collate


def conv_frames(lengths, kernels, strides):
    v = lengths.long()
    for k,s in zip(kernels,strides):
        v = torch.div(v-int(k),int(s),rounding_mode='floor')+1
    return v


class SimpleWavLMLarge(nn.Module):
    def __init__(self, backbone, n_classes):
        super().__init__()
        self.wavlm = backbone
        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(1024,n_classes)
        self.stage = 'frozen'

    def set_stage(self,stage):
        if stage not in ('frozen','finetune'):
            raise ValueError(stage)
        self.stage = stage
        self.wavlm.requires_grad_(False)
        if stage == 'finetune':
            for layer in self.wavlm.encoder.layers[18:24]:
                layer.requires_grad_(True)
        else:
            self.wavlm.eval()

    def train(self,mode=True):
        super().train(mode)
        if mode:
            if self.stage == 'frozen':
                self.wavlm.eval()
            else:
                self.wavlm.feature_extractor.eval()
                self.wavlm.feature_projection.eval()
                for layer in self.wavlm.encoder.layers[:18]:
                    layer.eval()
        return self

    def forward(self,values,mask):
        # No output_hidden_states: baseline does not pay for intermediate-layer fusion.
        with (torch.no_grad() if self.stage == 'frozen' else contextlib.nullcontext()):
            states = self.wavlm(values,attention_mask=mask,output_hidden_states=False,
                                return_dict=True).last_hidden_state
        lengths = conv_frames(mask.long().sum(-1),self.wavlm.config.conv_kernel,
                              self.wavlm.config.conv_stride)
        if (lengths <= 0).any() or (lengths > states.shape[1]).any():
            raise ValueError('Invalid WavLM frame-mask lengths')
        frame_mask = torch.arange(states.shape[1],device=states.device)[None,:] < lengths[:,None]
        count = frame_mask.sum(1,keepdim=True).clamp(min=1)
        pooled = (states.float()*frame_mask.unsqueeze(-1)).sum(1)/count.float()
        embeddings = self.norm(pooled.to(states.dtype))
        return self.classifier(self.dropout(embeddings)),embeddings


def make_optimizer(model, stage, a):
    groups = defaultdict(list)
    for name,p in model.named_parameters():
        if not p.requires_grad: continue
        if name.startswith('wavlm.encoder.layers.'):
            number = int(name.split('.')[3]); lr = a.wavlm_lr * (.8**(23-number))
        else:
            lr = a.head_lr if stage == 'frozen' else a.finetune_head_lr
        wd = .01 if p.ndim >= 2 and not name.endswith('bias') else 0.
        groups[(lr,wd)].append(p)
    return torch.optim.AdamW([{'params':ps,'lr':lr,'weight_decay':decay} for (lr,decay),ps in groups.items()],
                             betas=(.9,.999),eps=1e-8,foreach=True)


def make_schedule(opt, stage, a, updates):
    num_epochs = a.frozen_epochs if stage == 'frozen' else a.max_epochs-a.frozen_epochs
    total = max(1, num_epochs * updates)
    warmup = max(1,int(.10*total))
    def scale(step):
        if step < warmup:
            return max(.05,(step+1)/warmup)
        progress = min(1.,(step-warmup)/max(1,total-warmup))
        return .05+.95*(1+math.cos(math.pi*progress))/2
    return torch.optim.lr_scheduler.LambdaLR(opt,scale)


def model_state_cpu(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def atomic_torch_save(path, data):
    tmp = Path(str(path)+'.tmp')
    torch.save(data,tmp)
    os.replace(tmp,path)


def compute_metrics(frame, labels, bootstrap=5000, seed=42):
    y = frame.true_class.to_numpy(int)
    yhat = frame.pred_class.to_numpy(int)
    k = len(labels)
    p_cols = [f'p_{s}' for s in SLUGS]
    probs = frame[p_cols].to_numpy(float)
    p,r,f,s = precision_recall_fscore_support(y,yhat,labels=list(range(k)),zero_division=0)
    cm = confusion_matrix(y,yhat,labels=list(range(k)))
    speakers = []
    speaker_class = []
    for sid,sub in frame.groupby('subject_id'):
        yy = sub.true_class.to_numpy(int)
        yp = sub.pred_class.to_numpy(int)
        ps,rs,fs,ss = precision_recall_fscore_support(yy,yp,labels=list(range(k)),zero_division=0)
        speakers.append({'subject_id':sid,'n':len(sub),'accuracy':float(np.mean(yy==yp)),
                         'uar':float(np.mean(rs)),'macro_f1':float(np.mean(fs))})
        for c in range(k):
            speaker_class.append({'subject_id':sid,'emotion':labels[c],
                                  'class_id':c,'precision':ps[c],'recall':rs[c],'f1':fs[c],'support':ss[c]})
    sp = pd.DataFrame(speakers)
    pcs = pd.DataFrame(speaker_class)
    by_class = pd.DataFrame({'class_id':range(k),'emotion':labels,
                             'precision':p,'recall':r,'f1':f,'support':s})
    score = {'n_recordings':len(frame),'n_speakers':len(sp),
             'accuracy':float(np.mean(y==yhat)),'mean_subject_accuracy':float(sp.accuracy.mean()),
             'uar':float(np.mean(r)),'macro_f1':float(np.mean(f)),
             'weighted_f1':float(np.average(f,weights=s)),
             'cross_entropy':float(log_loss(y,probs,labels=list(range(k))))}
    try:
        score['auroc_macro_ovr']=float(roc_auc_score(y,probs,labels=list(range(k)),multi_class='ovr',average='macro'))
    except ValueError:
        score['auroc_macro_ovr']=None
    rng = np.random.default_rng(seed)
    if bootstrap > 0:
        samples = rng.choice(sp.accuracy.to_numpy(float),size=(bootstrap,len(sp)),replace=True).mean(1)
        score['mean_subject_accuracy_95ci']=[float(z) for z in np.quantile(samples,[.025,.975])]
    return score, sp, pcs, by_class, cm


def heatmap(matrix, xlabels, ylabels, title, path, normalized=False):
    fig,ax = plt.subplots(figsize=(max(6.,len(xlabels)*1.12),max(5.,len(ylabels)*.28+2)))
    im = ax.imshow(matrix,aspect='auto',cmap='Blues',vmin=0,vmax=1 if normalized else None)
    if len(ylabels) <= 9:
        for i in range(len(ylabels)):
            for j in range(len(xlabels)):
                val = matrix[i,j]
                ax.text(j,i,f'{val:.2f}' if normalized else str(int(val)),
                        ha='center',va='center',color='white' if im.norm(val)>.7 else 'black',fontsize=9)
    ax.set_xticks(np.arange(len(xlabels)),xlabels,rotation=35,ha='right')
    ax.set_yticks(np.arange(len(ylabels)),ylabels)
    ax.set(title=title,xlabel='Predicted emotion' if len(ylabels)==len(xlabels) else 'Emotion',
           ylabel='True emotion' if len(ylabels)==len(xlabels) else 'Speaker')
    fig.colorbar(im,ax=ax,shrink=.8)
    fig.tight_layout()
    fig.savefig(path.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(path.with_suffix('.svg'),bbox_inches='tight')
    plt.close(fig)


def export_evaluation(pred, folder, labels, checkpoint_epoch, label):
    folder.mkdir(parents=True,exist_ok=True)
    score,sp,pcs,by_class,cm = compute_metrics(pred,labels)
    score.update({'dataset':DATASET,'split':label,'best_epoch':int(checkpoint_epoch),
                  'unit':'original recording',
                  'speaker_overlap':True,'original_recording_disjoint':True})
    json_save(folder/'metrics.json',score)
    pred.to_csv(folder/'recording_predictions.csv',index=False)
    sp.to_csv(folder/'per_speaker_metrics.csv',index=False)
    pcs.to_csv(folder/'per_speaker_class_metrics.csv',index=False)
    by_class.to_csv(folder/'per_class_metrics.csv',index=False)
    pd.DataFrame(cm,index=labels,columns=labels).to_csv(folder/'confusion_counts.csv')
    norm = cm/np.maximum(cm.sum(axis=1,keepdims=True),1)
    pd.DataFrame(norm,index=labels,columns=labels).to_csv(folder/'confusion_normalized.csv')
    heatmap(cm,labels,labels,label.upper()+' confusion matrix',folder/'confusion_counts')
    heatmap(norm,labels,labels,label.upper()+' normalized confusion',folder/'confusion_normalized',True)
    for field in ('recall','f1'):
        pivot = pcs.pivot(index='subject_id',columns='class_id',values=field).reindex(index=sp.subject_id,columns=range(len(labels))).fillna(0)
        heatmap(pivot.to_numpy(),labels,[str(x) for x in sp.subject_id],
                f'{label.upper()} speaker by emotion {field}',folder/f'speaker_emotion_{field}',True)
    # Per-emotion precision/recall/F1.
    fig,ax=plt.subplots(figsize=(9,5))
    xx=np.arange(len(labels));width=.24
    for i,key in enumerate(['precision','recall','f1']):
        ax.bar(xx+(i-1)*width,by_class[key].to_numpy()*100,width,label=key.title())
    ax.set(ylim=(0,105),ylabel='Percent',title=label.upper()+' classwise metrics')
    ax.set_xticks(xx,labels);ax.legend();fig.tight_layout()
    fig.savefig(folder/'classwise_metrics.png',dpi=180);fig.savefig(folder/'classwise_metrics.svg');plt.close(fig)
    # Recording-level one-versus-rest ROC/PR plots.
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    y=pred.true_class.to_numpy(int)
    probs=pred[[f'p_{s}' for s in SLUGS]].to_numpy(float)
    fig,ax=plt.subplots(figsize=(7,6))
    fig2,ax2=plt.subplots(figsize=(7,6))
    for c,name in enumerate(labels):
        binary=(y==c).astype(int)
        if 0 < binary.sum() < len(binary):
            fp,tp,_=roc_curve(binary,probs[:,c]);ax.plot(fp,tp,label=f'{name}: {auc(fp,tp):.3f}')
            precision,recall,_=precision_recall_curve(binary,probs[:,c]);ax2.plot(recall,precision,label=name)
    ax.plot([0,1],[0,1],'--',color='gray',lw=1)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=f'{label.upper()} OvR ROC');ax.legend()
    ax2.set(xlabel='Recall',ylabel='Precision',title=f'{label.upper()} OvR precision-recall');ax2.legend()
    for f,basename in [(fig,'roc_ovr'),(fig2,'precision_recall_ovr')]:
        f.tight_layout();f.savefig(folder/f'{basename}.png',dpi=180);f.savefig(folder/f'{basename}.svg');plt.close(f)
    # Calibration analysis and misclassification confidence.
    confid=probs.max(1);correct=y==pred.pred_class.to_numpy(int)
    bins=np.linspace(0,1,11)
    rel=[]
    for i in range(len(bins)-1):
        mask=(confid>=bins[i]) & (confid <= bins[i+1] if i==9 else confid<bins[i+1])
        if mask.any():
            rel.append({'confidence':float(confid[mask].mean()),'accuracy':float(correct[mask].mean()),'n':int(mask.sum())})
    pd.DataFrame(rel).to_csv(folder/'reliability_bins.csv',index=False)
    fig,ax=plt.subplots(figsize=(6,6));ax.plot([0,1],[0,1],'--',color='gray')
    if rel:ax.plot([x['confidence'] for x in rel],[x['accuracy'] for x in rel],'o-')
    ax.set(xlim=(0,1),ylim=(0,1),xlabel='Confidence',ylabel='Accuracy',title=label.upper()+' reliability')
    fig.tight_layout();fig.savefig(folder/'reliability.png',dpi=180);fig.savefig(folder/'reliability.svg');plt.close(fig)
    return score


def plot_history(history, folder, frozen_epochs=DEFAULT_FROZEN_EPOCHS):
    if not history:return
    frame=pd.DataFrame(history)
    frame.to_csv(folder/'training_history.csv',index=False)
    for file,ys,title in [('training_curves',[('training_accuracy','Training crop accuracy'),('val_msa','Validation mean speaker accuracy'),('val_uar','Validation UAR')],'Training/validation performance'),
                          ('training_loss',[('training_ce','Training CE'),('val_cross_entropy','Validation CE')],'Training/validation loss')]:
        fig,ax=plt.subplots(figsize=(10,5))
        for col,name in ys:
            if col in frame:
                ax.plot(frame.epoch, frame[col].to_numpy()*(100 if file=='training_curves' else 1),marker='o',label=name)
        ax.axvline(frozen_epochs+.5,color='gray',ls='--',lw=1,label='Fine-tuning begins')
        ax.set(xlabel='Epoch',ylabel='Percent' if file=='training_curves' else 'Cross-entropy',title=title)
        ax.legend();ax.grid(alpha=.2)
        fig.tight_layout();fig.savefig(folder/f'{file}.png',dpi=180);fig.savefig(folder/f'{file}.svg');plt.close(fig)


def capture_rng():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'cpu':torch.get_rng_state(),
            'gpu':torch.cuda.get_rng_state_all()}


def restore_rng(data):
    random.setstate(data['python']);np.random.set_state(data['numpy']);torch.set_rng_state(data['cpu'])
    if data.get('gpu'):torch.cuda.set_rng_state_all(data['gpu'])


def make_checkpoint(model,epoch,stage,split_sha,config_sha,val,history,best_epoch,best_rank,stale,
                    opt=None,scheduler=None):
    d = {'checkpoint_format':'SIMPLE_WAVLM_LARGE_BASELINE_WITHIN_CORPUS_V1',
         'dataset':DATASET,'epoch':int(epoch),'stage':stage,
         'model':model_state_cpu(model),'split_sha':split_sha,'config_sha':config_sha,
         'best_epoch':best_epoch,'best_rank':best_rank,'stale':stale,
         'val_scores':val,'history':history,'rng':capture_rng()}
    if opt is not None:
        d['optimizer']=opt.state_dict();d['scheduler']=scheduler.state_dict()
    return d


def load_checkpoint(path,model,split_sha,config_sha,restore=False):
    ck=torch.load(path,map_location='cpu',weights_only=False)  # only load own trusted checkpoint
    for key,value in [('checkpoint_format','SIMPLE_WAVLM_LARGE_BASELINE_WITHIN_CORPUS_V1'),
                      ('dataset',DATASET),('split_sha',split_sha),('config_sha',config_sha)]:
        if ck.get(key)!=value:
            raise RuntimeError(f'Checkpoint incompatible: {key} mismatch in {path}')
    model.load_state_dict(ck['model'],strict=True)
    model.set_stage(ck['stage'])
    if restore:
        restore_rng(ck['rng'])
    return ck


def create_guarded_test_lock(test_dir, allow_crash_recovery, *, checkpoint_sha, split_sha):
    """Respect one-time final test; only retry a crashed attempt with zero outputs.

    A process SIGSEGV leaves final_test_started.lock in place. Because it never
    reaches output export, recovery is permissible only when nothing except the
    previous lock and earlier recovery records exists in the test folder.
    All recovered locks and reasons are saved for transparent paper auditing.
    """
    test_dir=Path(test_dir)
    test_dir.mkdir(parents=True,exist_ok=True)
    lock=test_dir/'final_test_started.lock'
    done=test_dir/'final_test_completed.json'
    if done.exists():
        raise RuntimeError('Final test already completed; a new test is forbidden.')
    permitted_names={'final_test_started.lock', 'crash_recovery_log.jsonl'}
    unexpected=[p for p in test_dir.iterdir() if
                p.name not in permitted_names and
                not p.name.startswith('final_test_started.crashed_')]
    if unexpected:
        raise RuntimeError('Refusing test (or crash recovery): existing test artifacts: '
                           + ', '.join(str(p.name) for p in unexpected[:12]))
    if lock.exists():
        if not allow_crash_recovery:
            raise RuntimeError('Previous final test attempt started. If it crashed with no '
                               'predictions/results, rerun with --recover-crashed-test.')
        if not lock.is_file():
            raise RuntimeError('Unexpected test lock type; will not recover')
        stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
        archived=test_dir/f'final_test_started.crashed_{stamp}.lock'
        os.replace(lock,archived)
        item={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'reason':'native crash prior to any persisted test output',
              'previous_lock':archived.name,
              'checkpoint_sha':checkpoint_sha,'split_sha':split_sha,
              'evaluation_runtime_only':{'workers':0,'pin_memory':False}}
        with open(test_dir/'crash_recovery_log.jsonl','a') as fd:
            fd.write(json.dumps(item,sort_keys=True)+'\n')
        print('[TEST RECOVERY] Old lock archived; zero prior test outputs found. '
              'Best checkpoint and dataset split remain verified.',flush=True)
    with open(lock,'x') as fd:
        fd.write(datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
    return lock


def score_quick(pred, labels):
    score,_,_,_,_=compute_metrics(pred,labels,bootstrap=0)
    return score


@torch.inference_mode()
def predict_recordings(model,frame,store,extractor,width,batch_size,workers,device,*,pin_memory=False):
    """Crash-resistant evaluation; recording-level views/averaging are unchanged.

    Default evaluation uses num_workers=0, pin_memory=False, synchronous H2D transfers.
    These options are *runtime* throughput controls and do not affect the trained model,
    validation selection, input crops, split, or checkpoint configuration checksum.
    """
    model.eval()
    print(f'[EVAL SAFE] recordings={len(frame)} workers={workers} batch={batch_size} '
          f'pin_memory={pin_memory}', flush=True)
    ds=EvaluationAudio(frame,store,width)
    loader=DataLoader(ds,batch_size=batch_size,shuffle=False,num_workers=workers,
                      pin_memory=pin_memory,collate_fn=make_collate(extractor,width),
                      **({'prefetch_factor':2} if workers else {}))
    sums={};counts=Counter()
    for b in tqdm(loader,desc=f'[EVAL] original recordings={len(frame)}',mininterval=1.,unit='batch'):
        with torch.autocast('cuda',dtype=torch.bfloat16):
            logits,_ = model(b['input_values'].to(device,non_blocking=pin_memory),
                             b['attention_mask'].to(device,non_blocking=pin_memory))
        values=logits.float().cpu().numpy()
        for j, rid in enumerate(b['recording_id']):
            if rid not in sums:sums[rid]=np.zeros(len(SLUGS),dtype=np.float64)
            sums[rid]+=values[j];counts[rid]+=1
    if len(sums)!=len(frame):
        raise RuntimeError('Evaluation is missing original recording predictions')
    frame=frame.sort_values('recording_id').reset_index(drop=True)
    z=np.stack([sums[rid]/counts[rid] for rid in frame.recording_id])
    probs=softmax(z,axis=1)
    pred=frame[['recording_id','subject_id','trial_id','emotion_id']].rename(columns={'emotion_id':'true_class'}).copy()
    pred['pred_class']=z.argmax(1);pred['correct']=(pred.true_class==pred.pred_class).astype(int)
    pred['n_windows']=[counts[rid] for rid in frame.recording_id]
    for cls,slug in enumerate(SLUGS):pred[f'p_{slug}']=probs[:,cls]
    return pred


def make_args():
    parser=argparse.ArgumentParser(description=f'{DATASET}: WavLM-Large simple baseline, original-recording-disjoint 70/15/15')
    parser.add_argument('--mode',choices=['self-test','audit','train','test','report'],default='audit')
    parser.add_argument('--data-root',type=Path,default=DEFAULT_DATA_ROOT)
    parser.add_argument('--output-dir',type=Path,default=None)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-epochs',type=int,default=DEFAULT_EPOCHS)
    parser.add_argument('--frozen-epochs',type=int,default=DEFAULT_FROZEN_EPOCHS)
    parser.add_argument('--patience',type=int,default=DEFAULT_PATIENCE)
    parser.add_argument('--min-delta',type=float,default=.002)
    parser.add_argument('--frozen-batch-size',type=int,default=DEFAULT_FROZEN_BATCH)
    parser.add_argument('--finetune-batch-size',type=int,default=DEFAULT_FINETUNE_BATCH)
    parser.add_argument('--eval-batch-size',type=int,default=DEFAULT_EVAL_BATCH)
    parser.add_argument('--effective-batch',type=int,default=DEFAULT_EFFECTIVE_BATCH)
    parser.add_argument('--workers',type=int,default=4,
                        help='Training DataLoader workers; kept at original default 4')
    parser.add_argument('--eval-workers',type=int,default=0,
                        help='Evaluation DataLoader workers; default 0 to avoid native multiprocessing faults')
    parser.add_argument('--eval-pin-memory',action='store_true',
                        help='Opt-in to pinned CPU memory during evaluation (off by default)')
    parser.add_argument('--recover-crashed-test',action='store_true',
                        help='Audit and recover a crashed test lock ONLY if no test output exists')
    parser.add_argument('--window-seconds',type=float,default=DEFAULT_WINDOW_SECONDS)
    parser.add_argument('--noise-probability',type=float,default=DEFAULT_NOISE)
    parser.add_argument('--head-lr',type=float,default=3e-4)
    parser.add_argument('--finetune-head-lr',type=float,default=1e-4)
    parser.add_argument('--wavlm-lr',type=float,default=5e-6)
    parser.add_argument('--label-smoothing',type=float,default=DEFAULT_LABEL_SMOOTHING)
    parser.add_argument('--checkpoint-revision',default='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c')
    parser.add_argument('--max-gpu-gib',type=float,default=13.5,
                        help='Advisory process memory target, not an absolute physical CUDA ceiling')
    parser.add_argument('--gpu-headroom-gib',type=float,default=1.5)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--unlock-test',action='store_true')
    parser.add_argument('--refresh-audit',action='store_true')
    parser.add_argument('--no-cache',action='store_true')
    parser.add_argument('--no-gradient-checkpointing',action='store_true')
    parser.add_argument('--reference-split',type=Path,default=None,
                        help='Optional Architecture A original-recording split CSV; refuse if labels/splits differ')
    parser.add_argument('--classes',choices=['four','seven'],default='seven') if DATASET=='pme4' else None
    a=parser.parse_args()
    if a.mode=='self-test':return a
    for batch in (a.frozen_batch_size,a.finetune_batch_size):
        if batch<=0 or batch%2 or a.effective_batch%batch:
            parser.error('Stage microbatches must be positive even divisors of effective-batch')
    if a.eval_batch_size<1 or a.workers<0 or a.eval_workers<0 or not (0<=a.noise_probability<=1):
        parser.error('Invalid dataloader settings')
    if a.recover_crashed_test and a.mode!='test':
        parser.error('--recover-crashed-test only permitted in test mode')
    if a.mode=='test' and not a.unlock_test:parser.error('Final test requires --unlock-test')
    if a.mode!='test' and a.unlock_test:parser.error('--unlock-test only valid in test mode')
    if a.frozen_epochs<1 or a.frozen_epochs>=a.max_epochs or a.patience<1:parser.error('Invalid training schedule')
    if not 1<=a.window_seconds<=10:parser.error('Window duration must be 1 to 10 seconds')
    return a


def check_reference_split(df, reference_path):
    path=Path(reference_path)
    if not path.exists():raise FileNotFoundError('Reference split not found: '+str(path))
    old=pd.read_csv(path)
    common=['recording_id','emotion_id','split']
    if not all(k in old.columns for k in common):
        raise ValueError(f'{path}: reference missing columns {common}')
    a=df[common].sort_values('recording_id').reset_index(drop=True)
    b=old[common].sort_values('recording_id').reset_index(drop=True)
    if not a.equals(b):
        raise RuntimeError(f'Recording labels/splits differ from Architecture A reference: {path}. Not a fair paired baseline.')
    print('[REFERENCE SPLIT MATCH] Identical recording labels and split assignments:',path,flush=True)


def make_paper_tables(out):
    """Produce exportable, recording-level paper summary once metrics are available."""
    paper=out/'paper_results';paper.mkdir(parents=True,exist_ok=True)
    rows=[];class_rows=[]
    for part in ('val','test'):
        f=out/part/'metrics.json'
        if not f.is_file():continue
        m=json.loads(f.read_text())
        rows.append({'dataset':DATASET,'split':part,'n_recordings':m['n_recordings'],
                     'n_speakers':m['n_speakers'],
                     'mean_subject_accuracy':m['mean_subject_accuracy'],
                     'msa_ci_lower':m['mean_subject_accuracy_95ci'][0],
                     'msa_ci_upper':m['mean_subject_accuracy_95ci'][1],
                     'pooled_accuracy':m['accuracy'],'uar':m['uar'],
                     'macro_f1':m['macro_f1'],'weighted_f1':m['weighted_f1'],
                     'auroc_ovr':m['auroc_macro_ovr'],
                     'best_epoch':m['best_epoch'],
                     'subject_overlap':True})
        per=pd.read_csv(out/part/'per_class_metrics.csv')
        per.insert(0,'split',part)
        class_rows.append(per)
    if rows:
        frame=pd.DataFrame(rows)
        frame.to_csv(paper/'table_1_summary_metrics.csv',index=False)
        frame.to_latex(paper/'table_1_summary_metrics.tex',index=False,float_format='%.4f')
    if class_rows:
        cls=pd.concat(class_rows,ignore_index=True)
        cls.to_csv(paper/'table_2_per_emotion_metrics.csv',index=False)
        cls.to_latex(paper/'table_2_per_emotion_metrics.tex',index=False,float_format='%.4f')


def self_test():
    x=torch.tensor([[1.,3.,3.,999.],[1.,1.,1.,1.]])
    mask=torch.tensor([[True,True,True,False],[True,True,True,True]])
    pooled=(x*mask).sum(1)/mask.sum(1)
    assert torch.allclose(pooled,torch.tensor([7/3,1.]))
    assert all_eval_starts(200,120)==[0,80]
    class Fake(nn.Module):
        def __init__(self):
            super().__init__()
            self.config=type('Config',(),{'conv_kernel':[2],'conv_stride':[2]})()
            self.encoder=type('Enc',(),{})()
            self.encoder.layers=nn.ModuleList([nn.Linear(4,4) for _ in range(24)])
            self.feature_extractor=nn.Identity()
            self.feature_projection=nn.Identity()
            self.weight=nn.Linear(4,1024)
        def forward(self,values,attention_mask,output_hidden_states=False,return_dict=True):
            n=values.shape[1]//2
            x=self.weight(values[:,:n*2].reshape(len(values),n,2).mean(-1).unsqueeze(-1).repeat(1,1,4))
            return type('Result',(),{'last_hidden_state':x})()
    dummy=SimpleWavLMLarge(Fake(),len(LABELS));dummy.set_stage('frozen')
    out,_=dummy(torch.ones(2,18),torch.tensor([[1]*18,[1]*12+[0]*6]))
    assert out.shape==(2,len(LABELS))
    out.sum().backward();assert dummy.classifier.weight.grad is not None
    # Split, class/speaker and batching smoke test from synthetic data.
    rows=[{'recording_id':f'r{s}_{c}_{i}','subject_id':s,'emotion_id':c} for s in range(4) for c in range(len(LABELS)) for i in range(3)]
    ds=type('DummyDS',(),{'rows':rows})()
    sampler=BalancedSpeakerPairs(ds,4,11,42)
    batches=list(iter(sampler));assert len(batches)==11 and all(len(x)==4 for x in batches)
    for batch in batches:
        groups=defaultdict(set)
        for j in batch:groups[rows[j]['emotion_id']].add(rows[j]['subject_id'])
        assert all(len(ids)>=2 for ids in groups.values())
    print('[SELF-TEST PASSED] model gradients, masks, evaluation views, paired sampler',flush=True)


def main():
    a=make_args()
    if a.mode=='self-test':
        self_test();return
    # Set PME4 labels before inspecting any data; EAV always uses its five full classes.
    global LABELS,SLUGS
    if DATASET=='pme4':
        LABELS = PME4_LABELS_FOUR if a.classes=='four' else PME4_LABELS_SEVEN
        SLUGS = tuple(x.lower() for x in LABELS)
    out=(a.output_dir if a.output_dir else OUTPUT_BASE / (a.classes+'class' if DATASET=='pme4' else 'fiveclass')).resolve()
    out.mkdir(parents=True,exist_ok=True)
    a.data_root=a.data_root.expanduser().resolve()
    frame=load_or_audit(a,out,LABELS)
    if a.reference_split:check_reference_split(frame,a.reference_split)
    if a.mode=='audit':return
    if a.mode=='report':
        for phase in ('val','test'):
            src=out/phase/'recording_predictions.csv'
            if src.is_file():
                score=export_evaluation(pd.read_csv(src),out/phase,LABELS,int(json.loads((out/'training_complete.json').read_text())['best_epoch']),phase)
                print('[REPORT]',phase,score,flush=True)
        make_paper_tables(out)
        return
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('A100-class GPU with CUDA BF16 is required for train/test')
    if not re.fullmatch(r'[a-f0-9]{40}',a.checkpoint_revision):
        raise ValueError('Checkpoint revision should be a pinned commit SHA')
    set_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    gpu=torch.device('cuda:0')
    total_bytes=torch.cuda.get_device_properties(0).total_memory
    alloc_budget=a.max_gpu_gib-a.gpu_headroom_gib
    if not .5 < alloc_budget < total_bytes/(1024**3):
        raise ValueError('Invalid GPU budget/headroom')
    torch.cuda.set_per_process_memory_fraction(alloc_budget*1024**3/total_bytes,device=0)
    print(f'[GPU] {torch.cuda.get_device_name(0)} allocation limit={alloc_budget:.2f}GiB / advisory target={a.max_gpu_gib:.2f}GiB',flush=True)
    from transformers import AutoFeatureExtractor, WavLMConfig, WavLMModel
    extractor=AutoFeatureExtractor.from_pretrained('microsoft/wavlm-large',revision=a.checkpoint_revision)
    conf=WavLMConfig.from_pretrained('microsoft/wavlm-large',revision=a.checkpoint_revision)
    if conf.hidden_size!=1024 or conf.num_hidden_layers!=24 or extractor.sampling_rate!=16000:
        raise ValueError('Unexpected WavLM-Large checkpoint dimensions/sample rate')
    conf.apply_spec_augment=False;conf.layerdrop=0.0
    backbone=WavLMModel.from_pretrained('microsoft/wavlm-large',revision=a.checkpoint_revision,config=conf)
    model=SimpleWavLMLarge(backbone,len(LABELS)).to(gpu)
    model.set_stage('frozen')
    if not a.no_gradient_checkpointing:
        try:model.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
        except TypeError:model.wavlm.gradient_checkpointing_enable()
    width=int(round(16000*a.window_seconds))
    store=AudioCache(out/'audio_cache',enabled=not a.no_cache)
    train=frame[frame.split=='train'].copy()
    val=frame[frame.split=='val'].copy()
    test=frame[frame.split=='test'].copy()
    ds=TrainingAudio(train,store,width,a.seed,a.noise_probability)
    updates=math.ceil(len(train)/a.effective_batch)
    split_sha=fingerprint(frame)
    settings={'data_root':str(a.data_root),'classes':list(LABELS),'seed':a.seed,
              'window_seconds':a.window_seconds,'noise_probability':a.noise_probability,
              'effective_batch':a.effective_batch,'updates_per_epoch':updates,
              'frozen_batch':a.frozen_batch_size,'finetune_batch':a.finetune_batch_size,
              'eval_batch':a.eval_batch_size,'frozen_epochs':a.frozen_epochs,
              'max_epochs':a.max_epochs,'patience':a.patience,'min_delta':a.min_delta,
              'head_lr':a.head_lr,'finetune_head_lr':a.finetune_head_lr,'wavlm_lr':a.wavlm_lr,
              'label_smoothing':a.label_smoothing,'checkpoint_revision':a.checkpoint_revision,
              'gradient_checkpointing':not a.no_gradient_checkpointing}
    cfg_sha=hashlib.sha256(json.dumps(settings,sort_keys=True).encode()).hexdigest()
    json_save(out/'resolved_config.json',{'dataset':DATASET,'protocol':PROTOCOL,
              'model':'WavLM-Large final-layer masked mean + LayerNorm + Dropout(0.30) + Linear(num_classes)',
              'no_ghf':True,'no_dptp':True,'no_supcon':True,
              'splits':frame.split.value_counts().to_dict(),'split_sha':split_sha,
              'settings_sha':cfg_sha,'settings':settings,
              'caveat':'All subjects overlap by design. Recording-level 70/15/15, not speaker-independent.'})
    print(f'[RUN] corpus={DATASET} labels={list(LABELS)} train={len(train)} val={len(val)} test={len(test)} '
          f'updates/epoch={updates} width={a.window_seconds}s',flush=True)
    if a.mode=='test':
        complete=out/'training_complete.json'
        if not complete.exists():raise RuntimeError('Training has not completed; test is embargoed')
        info=json.loads(complete.read_text())
        if info['split_sha']!=split_sha or info['config_sha']!=cfg_sha:
            raise RuntimeError('Training/test settings or split differ')
        if sha_file(out/'best.pth')!=info['best_checkpoint_sha']:
            raise RuntimeError('Best checkpoint has changed since training')
        best=load_checkpoint(out/'best.pth',model,split_sha,cfg_sha)
        create_guarded_test_lock(out/'test',a.recover_crashed_test,
                                 checkpoint_sha=info['best_checkpoint_sha'],split_sha=split_sha)
        pred=predict_recordings(model,test,store,extractor,width,
                                a.eval_batch_size,a.eval_workers,gpu,
                                pin_memory=a.eval_pin_memory)
        metrics=export_evaluation(pred,out/'test',LABELS,best['epoch'],'test')
        json_save(out/'test'/'final_test_completed.json',{
            'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'checkpoint_sha':sha_file(out/'best.pth'),'split_sha':split_sha,'best_epoch':best['epoch'],
            'final_test_evaluated_once':True,'metrics':metrics})
        make_paper_tables(out)
        print('[FINAL TEST]',metrics,flush=True)
        return
    if a.mode!='train':raise ValueError(a.mode)
    if (out/'training_complete.json').exists():
        raise RuntimeError('Already trained in this output dir. Choose a new output dir; do not tune on test.')
    start_epoch=1;best_epoch=0;best_rank=None;stale=0;history=[];previous=None
    if a.resume:
        previous=load_checkpoint(out/'last.pth',model,split_sha,cfg_sha,restore=True)
        start_epoch=int(previous['epoch'])+1
        best_epoch=previous['best_epoch'];best_rank=previous['best_rank']
        stale=previous['stale'];history=previous['history']
        print(f'[RESUME] epoch {start_epoch} (last completed: {start_epoch-1})',flush=True)
    if start_epoch>a.max_epochs:
        raise RuntimeError('All requested epochs finished; use the previous completed training receipt')
    prev_stage=None;opt=None;sched=None
    for epoch in range(start_epoch,a.max_epochs+1):
        stage='frozen' if epoch<=a.frozen_epochs else 'finetune'
        if stage!=prev_stage:
            model.set_stage(stage)
            opt=make_optimizer(model,stage,a)
            sched=make_schedule(opt,stage,a,updates)
            if previous is not None and previous['stage']==stage:
                opt.load_state_dict(previous['optimizer'])
                sched.load_state_dict(previous['scheduler'])
                # torch.load(map_location='cpu') may leave AdamW state on CPU.
                for group in opt.state.values():
                    for k,value in group.items():
                        if torch.is_tensor(value) and k!='step':group[k]=value.to(gpu)
            previous=None;prev_stage=stage
            print(f'[STAGE] {stage} trainable={sum(p.numel() for p in model.parameters() if p.requires_grad):,}',flush=True)
        batch_size=a.frozen_batch_size if stage=='frozen' else a.finetune_batch_size
        accumulation=a.effective_batch//batch_size
        ds.epoch=epoch-1
        sampler=BalancedSpeakerPairs(ds,batch_size,updates*accumulation,a.seed)
        sampler.epoch=epoch-1
        loader=DataLoader(ds,batch_sampler=sampler,num_workers=a.workers,pin_memory=True,
                          collate_fn=make_collate(extractor,width),
                          **({'prefetch_factor':2} if a.workers else {}))
        model.train();opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(gpu)
        total_loss=torch.zeros((),device=gpu,dtype=torch.float64)
        total_correct=torch.zeros((),device=gpu,dtype=torch.int64)
        count=0;steps=0
        t0=time.perf_counter()
        for i,b in enumerate(tqdm(loader,desc=f'[{DATASET} epoch {epoch}/{a.max_epochs} {stage}]',mininterval=1.,unit='batch')):
            yy=b['y'].to(gpu,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                logits,_=model(b['input_values'].to(gpu,non_blocking=True),
                               b['attention_mask'].to(gpu,non_blocking=True))
            ce=F.cross_entropy(logits.float(),yy,label_smoothing=a.label_smoothing)
            if not bool(torch.isfinite(ce)):
                raise FloatingPointError('Nonfinite train loss')
            (ce/accumulation).backward()
            total_loss+=ce.detach().double()*len(yy)
            total_correct+=(logits.detach().argmax(1)==yy).sum()
            count+=len(yy)
            if (i+1)%accumulation==0:
                clip=nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
                if not bool(torch.isfinite(clip)):
                    raise FloatingPointError('Nonfinite gradient norm')
                opt.step();sched.step();opt.zero_grad(set_to_none=True)
                steps+=1
        if steps != updates or count != updates*a.effective_batch:
            raise RuntimeError(f'Incorrect epoch coverage: {steps} optimizer steps, {count} examples')
        elapsed=time.perf_counter()-t0
        train_loss=float(total_loss.item()/count)
        train_acc=float(total_correct.item()/count)
        pred=predict_recordings(model,val,store,extractor,width,
                                a.eval_batch_size,a.eval_workers,gpu,
                                pin_memory=a.eval_pin_memory)
        metric=score_quick(pred,LABELS)
        row={'epoch':epoch,'stage':stage,'training_ce':train_loss,
             'training_accuracy':train_acc,'val_msa':metric['mean_subject_accuracy'],
             'val_uar':metric['uar'],'val_macro_f1':metric['macro_f1'],
             'val_cross_entropy':metric['cross_entropy'],
             'epoch_seconds':elapsed,'peak_allocated_gib':torch.cuda.max_memory_allocated(gpu)/2**30}
        history.append(row)
        plot_history(history,out,a.frozen_epochs)
        rank=(metric['mean_subject_accuracy'],metric['uar'],metric['macro_f1'],-metric['cross_entropy'],-epoch)
        if best_rank is None or rank > tuple(best_rank):
            best_epoch=epoch;best_rank=list(rank);stale=0
            atomic_torch_save(out/'best.pth',make_checkpoint(model,epoch,stage,split_sha,cfg_sha,metric,
                        history,best_epoch,best_rank,stale))
            pred.to_csv(out/'best_validation_predictions.csv',index=False)
            print(f'[BEST] epoch {epoch} MSA={metric["mean_subject_accuracy"]:.2%}',flush=True)
        elif stage=='finetune' and metric['mean_subject_accuracy']<=float(best_rank[0])+a.min_delta:
            stale+=1
        atomic_torch_save(out/'last.pth',make_checkpoint(model,epoch,stage,split_sha,cfg_sha,metric,
                    history,best_epoch,best_rank,stale,opt,sched))
        print(f'[EPOCH {epoch}] train_acc={train_acc:.2%} train_ce={train_loss:.4f} '
              f'val_msa={metric["mean_subject_accuracy"]:.2%} val_uar={metric["uar"]:.2%} '
              f'val_f1={metric["macro_f1"]:.2%} peak_alloc={row["peak_allocated_gib"]:.2f}GiB '
              f'seconds={elapsed:.1f} best={best_epoch} patience={stale}/{a.patience}',flush=True)
        if stage=='finetune' and stale>=a.patience:
            print('[EARLY STOP] validation MSA no longer improving',flush=True)
            break
    best=load_checkpoint(out/'best.pth',model,split_sha,cfg_sha)
    best_pred=predict_recordings(model,val,store,extractor,width,
                                      a.eval_batch_size,a.eval_workers,gpu,
                                      pin_memory=a.eval_pin_memory)
    metrics=export_evaluation(best_pred,out/'val',LABELS,best['epoch'],'val')
    json_save(out/'training_complete.json',{'best_epoch':best['epoch'],'best_val_metrics':metrics,
              'split_sha':split_sha,'config_sha':cfg_sha,
              'best_checkpoint_sha':sha_file(out/'best.pth'),'test_locked':True,
              'best_epoch_selection':'validation mean subject accuracy, UAR as tie breaker',
              'speaker_independent':False})
    make_paper_tables(out)
    print(f'[TRAINING COMPLETE] best epoch={best["epoch"]} val MSA={metrics["mean_subject_accuracy"]:.2%}; test LOCKED.',flush=True)


if __name__=='__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('[INTERRUPTED] Use --mode train --resume to continue from last completed epoch',file=sys.stderr)
        raise SystemExit(130)
