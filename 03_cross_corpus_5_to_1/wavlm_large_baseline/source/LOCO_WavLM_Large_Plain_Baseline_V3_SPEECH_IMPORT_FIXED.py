#!/usr/bin/env python3
"""Six-corpus Leave-One-Corpus-Out (LOCO) speech-emotion baseline.

MODEL: microsoft/wavlm-large (pinned revision) + masked temporal mean pooling
       + LayerNorm + Dropout + one Linear(1024 -> 4) classifier.
       NO GHF, DPTP, adapters, audio-language models, ensemble, or extra encoders.

CORPORA (dataset root): EAV, CREMA-D, EmoDB, EMOVO, PME4, RAVDESS.
Only four emotion classes: Neutral(0), Sad(1), Angry(2), Happy(3).

Each of six independently initialized folds trains on FIVE source corpora,
validates on SOURCE speaker-disjoint subsets only, and tests on ALL clips from
its sixth, completely UNSEEN target corpus. There is no target adaptation.

V2 FIX: audit decodes/resamples every candidate before locking the six folds,
quarantining silent/unusable WAVs; --rebuild-audit is safe only before training.
V3 FIX: protects this speech-only process from an incompatible TorchVision
optional dependency in Transformers; no dataset or model protocol changes.

All generated files including HF weights cache are under --project-root.
Data is read only from --data-root. No source WAVs are modified.

Example:
    python LOCO_WavLM_Large_Plain_Baseline.py --mode smoke
    python LOCO_WavLM_Large_Plain_Baseline.py --mode audit
    python LOCO_WavLM_Large_Plain_Baseline.py --mode prepare
    python -u LOCO_WavLM_Large_Plain_Baseline.py --mode preflight
    python -u LOCO_WavLM_Large_Plain_Baseline.py --mode train --target all
    python -u LOCO_WavLM_Large_Plain_Baseline.py --mode test --target all --unlock-test

A100 recommended: BF16 AMP, 4s crops, gradient checkpointing, frozen warm-up,
then finetune upper 6 WavLM blocks. Reinitialize model in every fold.

IMPORTANT: A target is NEVER inspected for validation/epoch selection. Source
validation speakers are disjoint from source training speakers within corpus.
Reported external test results reflect different datasets and languages, not
only unseen speakers. Source recordings may share scripted prompts, a limitation
for interpretation. Every fold needs independent model training.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import dataclasses
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time
import torch
from torch import nn
from collections import Counter, defaultdict, OrderedDict
from dataclasses import dataclass
from pathlib import Path

# Only standard libs at import time. Keep any pretrained hub cache INSIDE project.
PROJECT_DEFAULT = Path('/workspace/avyakt-sc-final/final-speech-rp')
DATA_DEFAULT = Path('/workspace/avyakt-sc-final/data')
CORPORA = ('EAV', 'CREMA-D', 'EmoDB', 'EMOVO', 'PME4', 'RAVDESS')
FOLD_NAMES = {n: n.lower().replace('-', '_') for n in CORPORA}
CLASS_NAMES = ('Neutral', 'Sad', 'Angry', 'Happy')
CLASS_TO_ID = {c:i for i,c in enumerate(CLASS_NAMES)}
MODEL_ID = 'microsoft/wavlm-large'
MODEL_REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
EXPECTED_COUNTS = {'EAV':3360, 'CREMA-D':4900, 'EmoDB':339,
                   'EMOVO':336, 'PME4':2189, 'RAVDESS':672}
AUDIT_VERSION = 'loco_plain_wavlm_v2_signal_audit_20261108'


def setup_modules(project_root):
    # Must be set before transformers / huggingface_hub is imported.
    cache = project_root / 'hf_cache'
    cache.mkdir(parents=True, exist_ok=True)
    os.environ['HF_HOME'] = str(cache)
    os.environ['HF_HUB_CACHE'] = str(cache / 'hub')
    global np, pd, sf, signal, plt
    import numpy as np
    import pandas as pd
    import soundfile as sf
    from scipy import signal
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    from torch import nn
    if int(np.__version__.split('.')[0]) >= 2:
        print('[WARNING] NumPy>=2 is installed. For this CUDA container, NumPy 1.26.4 is recommended.',flush=True)


def log(*items):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), '|', *items, flush=True)


def atomic_json(path, value):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix(path.suffix+'.writing')
    temp.write_text(json.dumps(value,indent=2,default=json_default),encoding='utf8')
    temp.replace(path)


def json_default(v):
    if hasattr(v,'item'): return v.item()
    if isinstance(v,(Path,)): return str(v)
    raise TypeError(type(v).__name__)


def sha_file(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def digest_json(obj):
    return hashlib.sha256(json.dumps(obj,sort_keys=True,default=json_default).encode()).hexdigest()


def emotion_name(text):
    s=re.sub('[^a-z]','',str(text).lower())
    return {'neutral':'Neutral','neu':'Neutral','sad':'Sad','sadness':'Sad',
            'tri':'Sad','angry':'Angry','anger':'Angry','rab':'Angry',
            'happy':'Happy','happiness':'Happy','gio':'Happy'}.get(s)


def row(dataset, path, speaker, emotion, trial_key, speech_start=None, speech_end=None):
    return {'dataset':dataset,'file_path':str(path.resolve()),
            'speaker_id':f'{dataset}:{speaker}', 'emotion':emotion,
            'class_id':CLASS_TO_ID[emotion], 'recording_id':f'{dataset}/{trial_key}',
            'speech_start':speech_start,'speech_end':speech_end}


def wavs_recursive(root):
    # use pathlib.rglob, robust case-insensitive suffix
    return sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower()=='.wav')


def scan_eav(root):
    re_eav=re.compile(r'^(\d+)_Trial_\d+_Speaking_([A-Za-z]+)_aud\.wav$',re.I)
    for d in sorted(root.glob('subject*')):
        if not re.fullmatch(r'subject\d+',d.name,re.I): continue
        sid=re.search(r'\d+',d.name).group()
        for p in sorted((d/'Audio').glob('*')):
            if not p.is_file() or p.suffix.lower()!='.wav':continue
            m=re_eav.match(p.name)
            if not m:
                raise RuntimeError(f'Unrecognized EAV speech WAV filename: {p}')
            emo=emotion_name(m.group(2))
            if emo:yield row('EAV',p,f's{int(sid):02d}',emo,f's{int(sid):02d}_event{int(m.group(1)):03d}')


def scan_crema(root):
    base=root/'AudioWAV'
    if not base.is_dir(): raise FileNotFoundError(f'CREMA-D AudioWAV missing: {base}')
    code={'ANG':'Angry','HAP':'Happy','SAD':'Sad','NEU':'Neutral'}
    for p in wavs_recursive(base):
        parts=p.stem.upper().split('_')
        if len(parts)<4 or not re.fullmatch(r'\d{4}',parts[0]):
            raise RuntimeError(f'Unrecognized CREMA-D WAV: {p}')
        if parts[2] in code:yield row('CREMA-D',p,parts[0],code[parts[2]],p.stem)


def scan_emodb(root):
    base=root/'wav'
    if not base.is_dir(): raise FileNotFoundError(f'EmoDB wav missing: {base}')
    codes={'N':'Neutral','T':'Sad','W':'Angry','F':'Happy'}
    for p in wavs_recursive(base):
        m=re.fullmatch(r'(\d{2})[a-z]\d{2}([a-z])[a-z]?',p.stem,re.I)
        if not m:raise RuntimeError(f'Unrecognized EmoDB WAV: {p}')
        emo=codes.get(m.group(2).upper())
        if emo:yield row('EmoDB',p,m.group(1),emo,p.stem)


def scan_emovo(root):
    base=root/'EMOVO'
    if not base.is_dir():raise FileNotFoundError(f'EMOVO extracted dir missing: {base}')
    for p in wavs_recursive(base):
        m=re.fullmatch(r'([a-z]+)-([fm]\d+)-(.+)',p.stem,re.I)
        if not m:raise RuntimeError(f'Unrecognized EMOVO WAV: {p}')
        emo=emotion_name(m.group(1))
        if emo:yield row('EMOVO',p,m.group(2).lower(),emo,p.stem)


def scan_pme4(root):
    meta=root/'PME4_dataset_configs.csv'
    if not meta.is_file():raise FileNotFoundError(f'PME4 label metadata missing: {meta}')
    seen=set()
    with meta.open(encoding='utf-8-sig',newline='') as f:
        for linenum,raw in enumerate(csv.DictReader(f),start=2):
            r={str(k).strip().lower():v for k,v in raw.items() if k is not None}
            try:sid=int(float(r['subject']));trial=int(float(r['trial']))
            except Exception as exc:raise RuntimeError(f'Bad PME4 CSV line {linenum}: {r}') from exc
            key=(sid,trial)
            if key in seen:raise RuntimeError(f'Duplicate PME4 config row subject/trial {key}')
            seen.add(key)
            emo=emotion_name(r.get('emotion',''))
            if emo is None:continue
            prefix=f's{sid:02d}_t{trial:03d}'
            p1=root/f's{sid:02d}'/f's{sid:02d}'/f't{trial:03d}'/f'{prefix}.wav'
            p2=root/f's{sid:02d}'/f't{trial:03d}'/f'{prefix}.wav'
            p=p1 if p1.is_file() else p2
            if not p.is_file():
                log('[MISSING PME4 WAV]',p1)
                continue
            def val(s):
                try:return float(r.get(s,''))
                except (TypeError,ValueError):return None
            yield row('PME4',p,f's{sid:02d}',emo,prefix,
                      speech_start=val('speech_start_time'),speech_end=val('speech_stop_time'))


def scan_ravdess(root):
    found=defaultdict(list)
    for p in wavs_recursive(root):found[p.name.lower()].append(p)
    class_map={'01':'Neutral','03':'Happy','04':'Sad','05':'Angry'}
    for lower,paths in sorted(found.items()):
        parts=Path(lower).stem.split('-')
        if len(parts)!=7 or not all(re.fullmatch(r'\d{2}',x) for x in parts):
            raise RuntimeError(f'Unrecognized RAVDESS WAV filename: {paths[0]}')
        if parts[0]!='03' or parts[1]!='01' or parts[2] not in class_map:continue
        paths=sorted(paths,key=lambda p:(len(p.relative_to(root).parts),str(p)))
        if len(paths)>1 and len({sha_file(x) for x in paths})>1:
            raise RuntimeError(f'RAVDESS same filename has DIFFERENT WAV bytes across copies: {paths}. '
                               'Fix extraction before a research experiment.')
        yield row('RAVDESS',paths[0],parts[6],class_map[parts[2]],Path(lower).stem)


def decode_and_preprocess_wave(path, dataset, speech_start, speech_end, args):
    """Single quality/preprocessing contract shared by AUDIT and PREPARE.

    A header-valid WAV can still decode to silence. Both stages MUST enforce
    identical checks before an immutable six-fold manifest is accepted.
    Returns (16 kHz mono float32 waveform, trim_applied, trim_metadata_fallback).
    """
    wave, sr = sf.read(str(path), dtype='float32', always_2d=True)
    if sr < 8000 or wave.ndim != 2 or len(wave) == 0:
        raise ValueError('empty/invalid decoded waveform')
    if not np.isfinite(wave).all():
        raise ValueError('NaN/Inf in decoded audio')
    wave = wave.mean(axis=1)
    trimmed = False
    fallback = False
    if args.pme4_speech_trim and dataset == 'PME4':
        try:
            st, ed = float(speech_start), float(speech_end)
        except (TypeError, ValueError):
            st, ed = float('nan'), float('nan')
        if np.isfinite(st) and np.isfinite(ed) and 0 <= st < ed <= len(wave) / sr + 0.05 and ed - st >= .12:
            lo = max(0, int((st - args.trim_context) * sr))
            hi = min(len(wave), int((ed + args.trim_context) * sr))
            wave = wave[lo:hi]
            trimmed = True
        else:
            fallback = True
    if sr != 16000:
        gcd = math.gcd(int(sr), 16000)
        wave = signal.resample_poly(wave, 16000 // gcd, int(sr) // gcd)
    wave = np.asarray(wave, dtype=np.float32)
    if len(wave) < int(args.min_audio_seconds * 16000):
        raise ValueError('too short after trim/resampling')
    if not np.isfinite(wave).all():
        raise ValueError('NaN/Inf after resampling')
    if float(np.max(np.abs(wave))) < 1e-7:
        raise ValueError('invalid/silent resampled audio')
    return wave, trimmed, fallback


def discover(args):
    root=Path(args.data_root).resolve()
    roots={'EAV':root/'EAV','CREMA-D':root/'crema-d','EmoDB':root/'emodb',
           'EMOVO':root/'emovo','PME4':root/'pme4'/'pme4-dataset','RAVDESS':root/'ravdess'}
    scanners={'EAV':scan_eav,'CREMA-D':scan_crema,'EmoDB':scan_emodb,
              'EMOVO':scan_emovo,'PME4':scan_pme4,'RAVDESS':scan_ravdess}
    rows=[]
    for corpus in CORPORA:
        if not roots[corpus].is_dir():raise FileNotFoundError(f'{corpus} missing: {roots[corpus]}')
        got=list(scanners[corpus](roots[corpus]));rows+=got
        log('[DISCOVERY]',corpus, 'eligible labeled WAV paths:',len(got))
    if len(rows)!=len({r['recording_id'] for r in rows}):
        raise RuntimeError('Duplicate recording IDs detected. Stop and resolve.')
    df=pd.DataFrame(rows)
    if df.empty:raise RuntimeError('No eligible audio')
    basic=[]
    for i,r in enumerate(df.itertuples(index=False)):
        p=Path(r.file_path)
        try:
            inf=sf.info(str(p))
            if inf.duration < args.min_audio_seconds or inf.samplerate<8000 or inf.channels<1:
                raise ValueError(f'Invalid header duration={inf.duration:.4f}, rate={inf.samplerate}, channels={inf.channels}')
            basic.append(dict(duration_sec=float(inf.duration),sample_rate=int(inf.samplerate),channels=int(inf.channels),
                              file_bytes=p.stat().st_size, file_mtime_ns=p.stat().st_mtime_ns, status='usable',sha256=''))
        except Exception as exc:
            basic.append(dict(duration_sec=0.,sample_rate=0,channels=0,file_bytes=0,
                              status='invalid_wav_header',sha256='',file_mtime_ns=0,issue=str(exc)[:300]))
        if (i+1)%2000==0:log('[HEADERS]',i+1,'/',len(df))
    more=pd.DataFrame(basic)
    df=pd.concat([df.reset_index(drop=True),more],axis=1)
    df.loc[df.status!='usable',['emotion']] = df.loc[df.status!='usable',['emotion']]
    if not args.skip_sha:
        usable_idxs=df.index[df.status=='usable'].tolist()
        for j,i in enumerate(usable_idxs):
            df.at[i,'sha256']=sha_file(df.at[i,'file_path'])
            if (j+1)%2000==0:log('[HASHES]',j+1,'/',len(usable_idxs))
        for hash_value,g in df[df.status=='usable'].groupby('sha256',sort=True):
            if len(g)<2:continue
            # quarantine all members, not just one; avoids cross-corpus leakage
            df.loc[g.index,'status']='duplicate_quarantined'
            df.loc[g.index,'issue']=f'exact duplicate WAV bytes in {len(g)} recording IDs (sha256={hash_value})'
    # Check decoded waveform integrity BEFORE locking any fold splits. Header
    # parsing and WAV SHA checks cannot detect correctly encoded silent audio.
    # This deliberately uses the SAME decoder/trim/resampler as prepare_audio.
    candidate_indices = df.index[df.status == 'usable'].tolist()
    bad_signal = 0
    for j, i in enumerate(candidate_indices, 1):
        r = df.loc[i]
        try:
            waveform, _, _ = decode_and_preprocess_wave(
                r.file_path, r.dataset, r.speech_start, r.speech_end, args)
            del waveform
        except Exception as exc:
            df.at[i, 'status'] = 'invalid_audio_signal'
            df.at[i, 'issue'] = str(exc)[:500]
            bad_signal += 1
            log('[AUDIO QA QUARANTINE]', r.recording_id, str(exc))
        if j % 1000 == 0:
            log('[AUDIO QA]', j, '/', len(candidate_indices),
                'signal-invalid so far:', bad_signal)
    log('[AUDIO QA COMPLETE]', len(candidate_indices), 'inspected,',
        bad_signal, 'invalid decoded signals quarantined before splitting')
    counts=df[df.status=='usable'].dataset.value_counts().to_dict()
    for corpus in CORPORA:
        expect=EXPECTED_COUNTS[corpus]
        actual=int(counts.get(corpus,0))
        if actual!=expect:log('[NOTE]',corpus,'expected audit count',expect,'actual usable',actual,
                              '(may differ because of quarantine/missing or extraction)')
    if set(counts)!=set(CORPORA):raise RuntimeError('Missing eligible corpus after audio quality / duplicate quarantine')
    return df


def choose_source_val_speakers(df,seed):
    # Subject-level split is stable globally, independent of which target is omitted.
    rows=[]
    for corpus in CORPORA:
        g=df[(df.dataset==corpus)&(df.status=='usable')]
        subjects=sorted(g.speaker_id.unique().tolist())
        nval=max(1,min(len(subjects)-1,int(round(.15*len(subjects)))))
        rng=np.random.default_rng(seed+int(hashlib.sha256(corpus.encode()).hexdigest()[:8],16))
        sel=None
        for attempt in range(2000):
            trial=sorted(rng.choice(subjects,size=nval,replace=False).tolist())
            val=g[g.speaker_id.isin(trial)]
            train=g[~g.speaker_id.isin(trial)]
            if set(val.class_id.unique())==set(range(4)) and set(train.class_id.unique())==set(range(4)):
                sel=set(trial);break
        if sel is None:raise RuntimeError(f'Cannot make 4-class speaker-disjoint val on {corpus} ({nval} val subjects).')
        for speaker in subjects: rows.append({'dataset':corpus,'speaker_id':speaker,
                                             'source_role':'val' if speaker in sel else 'train'})
    return pd.DataFrame(rows)


def make_fold_manifest(audit,source_speaker_split,target):
    good=audit[audit.status=='usable'].copy()
    roles=dict(zip(source_speaker_split.speaker_id,source_speaker_split.source_role))
    good['split']=['test' if dataset==target else roles[speaker]
                    for dataset,speaker in zip(good.dataset,good.speaker_id)]
    for corpus in CORPORA:
        part=good[good.dataset==corpus]
        assigned=set(part.split)
        if corpus==target:
            if assigned!={'test'}:raise AssertionError(f'Target {corpus} not exclusively test')
        elif assigned!={'train','val'}:raise AssertionError(f'Source {corpus} missing train or val')
    train=good[good.split=='train'];val=good[good.split=='val'];test=good[good.split=='test']
    for a,b in ((train,val),(train,test),(val,test)):
        assert not (set(a.recording_id)&set(b.recording_id))
    assert not (set(train.speaker_id)&set(val.speaker_id))
    assert set(test.dataset)=={target}
    assert target not in set(train.dataset) and target not in set(val.dataset)
    if 'sha256' in good.columns and good.sha256.notna().any() and good.sha256.ne('').any():
        for a,b in ((train,val),(train,test),(val,test)):
            assert not (set(a.sha256)&set(b.sha256))
    return good.sort_values(['dataset','speaker_id','recording_id']).reset_index(drop=True)


class OutputTee:
    def __init__(self,terminal,log_file):
        self.terminal=terminal
        self.log_file=log_file
    def write(self,text):
        self.terminal.write(text)
        self.log_file.write(text)
    def flush(self):
        self.terminal.flush()
        self.log_file.flush()
    def isatty(self):return bool(self.terminal.isatty())
    def fileno(self):return self.terminal.fileno()


def enable_logging(project,args):
    dest=project/'logs';dest.mkdir(exist_ok=True)
    filename=f"{args.mode}_{FOLD_NAMES.get(args.target,args.target)}_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}.log"
    logfile=open(dest/filename,'a',encoding='utf8',buffering=1)
    sys.stdout=OutputTee(sys.stdout,logfile)
    log('[LOG FILE]',dest/filename)


def safe_setup(args):
    project=Path(args.project_root).expanduser().resolve()
    source=Path(args.data_root).expanduser().resolve()
    if project==source or source in project.parents:
        raise RuntimeError('Project/output root cannot be inside source dataset tree')
    project.mkdir(parents=True,exist_ok=True)
    setup_modules(project)
    enable_logging(project,args)
    log('Project outputs:',project)
    log('Data read-only:',source)
    return project


def get_audit(args,project,force=False):
    ap=project/'audit'/'eligible_recordings.csv'
    sp=project/'audit'/'source_validation_speakers.csv'
    conf=project/'audit'/'audit_metadata.json'
    config={'data_root':str(Path(args.data_root).resolve()),'seed':args.seed,
            'skip_sha':args.skip_sha,'min_audio_seconds':args.min_audio_seconds,
            'pme4_speech_trim':bool(args.pme4_speech_trim),
            'trim_context':float(args.trim_context),
            'audit_version':AUDIT_VERSION}
    if ap.is_file() and sp.is_file() and conf.is_file() and not force:
        saved=json.loads(conf.read_text())
        if saved.get('audit_config') != config:
            raise RuntimeError('Existing audit configuration differs. Use --rebuild-audit, '
                               'or restore original flags; do not change split mid-experiment.')
        audit=pd.read_csv(ap,keep_default_na=False)
        for f in ('speech_start','speech_end','duration_sec'):
            if f in audit: audit[f]=pd.to_numeric(audit[f],errors='coerce')
        speakers=pd.read_csv(sp,keep_default_na=False)
        assert saved['audit_sha256']==digest_json(audit[['dataset','recording_id','speaker_id','status','file_path','file_bytes','file_mtime_ns','sha256']].to_dict('records'))
        return audit,speakers,saved['audit_sha256']
    if not force and (ap.exists()!=sp.exists()):raise RuntimeError('Partial audit files. Use --rebuild-audit')
    if force:
        # Do not change an experimental cohort once a model was trained,
        # including last.pth (interrupted) or any already tested fold.
        existing_models = [str(p) for pat in ('best.pth','last.pth','training_complete.json')
                           for p in (project/'folds').glob('*/'+pat)]
        if existing_models:
            raise RuntimeError('Refusing to rebuild split after training started: '+str(existing_models[:5])+
                               '. Use a NEW project root for a new cohort.')
        # Archive old audit/split manifests so exclusions are transparent in the paper.
        originals = ([p for p in (ap,sp,conf,
                                 project/'audit'/'preprocessing_errors.csv',
                                 project/'audit'/'excluded_or_duplicate_files.csv',
                                 project/'audit'/'class_counts_by_corpus.csv') if p.is_file()] +
                     list((project/'folds').glob('*/immutable_split_manifest.csv')))
        if originals:
            archive = project/'audit'/'superseded_before_signal_qa_v2'
            archive.mkdir(parents=True,exist_ok=True)
            for oldfile in originals:
                dst = archive / (oldfile.parent.name + '__' + oldfile.name)
                if not dst.exists(): shutil.copy2(oldfile,dst)
            log('[AUDIT BACKUP]',len(originals),'prior protocol files retained in',archive)
    (project/'audit').mkdir(parents=True,exist_ok=True)
    audit=discover(args)
    audit.to_csv(ap,index=False)
    report=audit[audit.status!='usable']
    report.to_csv(project/'audit'/'excluded_or_duplicate_files.csv',index=False)
    speakers=choose_source_val_speakers(audit,args.seed)
    speakers.to_csv(sp,index=False)
    good=audit[audit.status=='usable']
    counts=good.groupby(['dataset','emotion']).size().unstack(fill_value=0).reindex(index=CORPORA,columns=CLASS_NAMES,fill_value=0)
    counts.to_csv(project/'audit'/'class_counts_by_corpus.csv')
    sc=good.groupby('dataset').agg(n_recordings=('file_path','size'),n_speakers=('speaker_id','nunique'),
                                   total_hours=('duration_sec',lambda s:float(s.sum()/3600)))
    sc.to_csv(project/'audit'/'usable_corpus_summary.csv')
    sha=digest_json(audit[['dataset','recording_id','speaker_id','status','file_path','file_bytes','file_mtime_ns','sha256']].to_dict('records'))
    atomic_json(conf,dict(audit_config=config,audit_sha256=sha,usable_counts=good.dataset.value_counts().to_dict(),
                          classes=CLASS_NAMES,external_corpus_label_mapping='normalized four emotions',
                          exact_duplicate_policy='all members quarantined (when hashing enabled)',
                          source_validation_speakers='stable corpus-wise 15% group split; never target-used'))
    log('[AUDIT] Completed,',len(good),'valid four-class WAVs; SHA',sha[:16])
    for target in CORPORA:
        fold=make_fold_manifest(audit,speakers,target)
        folder=project/'folds'/FOLD_NAMES[target];folder.mkdir(parents=True,exist_ok=True)
        fold.to_csv(folder/'immutable_split_manifest.csv',index=False)
        log('[FOLD]',target,fold.split.value_counts().to_dict(),
            'target test subjects=',fold[fold.split=='test'].speaker_id.nunique())
    return audit,speakers,sha


def cache_settings(args):
    return dict(fs=16000,trim_pme4=bool(args.pme4_speech_trim),trim_context_sec=float(args.trim_context),
                preprocessing_version='mono_resample_poly_normalize_per_window_v1')


def cache_file_for(project,r,settings):
    token=digest_json(dict(path=r.file_path, size=int(r.file_bytes),
                           time_ns=Path(r.file_path).stat().st_mtime_ns,
                           speaker_id=r.speaker_id, settings=settings))
    return project/'cache'/'audio_16khz'/r.dataset/token[:2]/(token+'.npy')


def prepare_audio(args,project,audit):
    good=audit[audit.status=='usable']
    settings=cache_settings(args)
    meta=project/'cache'/'cache_config.json'
    if meta.is_file() and json.loads(meta.read_text())!=settings:
        raise RuntimeError('Cache preprocessing flags changed. Use a new project directory for a new experiment.')
    meta.parent.mkdir(parents=True,exist_ok=True)
    atomic_json(meta,settings)
    out=[];failed=[];used_trim=0;skipped_trim=0
    for j,r in enumerate(good.itertuples(index=False),start=1):
        path=cache_file_for(project,r,settings)
        trimmed=False
        if not path.is_file():
            try:
                wave, trimmed, fallback = decode_and_preprocess_wave(
                    r.file_path, r.dataset, r.speech_start, r.speech_end, args)
                if fallback: skipped_trim += 1
                path.parent.mkdir(parents=True,exist_ok=True)
                temp=path.with_name(path.stem+'.building.npy')
                np.save(temp,wave)
                os.replace(temp,path)
            except Exception as exc:
                failed.append(dict(recording_id=r.recording_id,file_path=r.file_path,error=str(exc)[:500]))
                log('[AUDIO ERROR]',r.recording_id,exc)
                continue
        else:
            # Existing cache from an interrupted prepare run is reusable, but
            # verify that it isn't corrupted or silent before trusting it.
            try:
                cached=np.load(path,mmap_mode='r',allow_pickle=False)
                if cached.ndim!=1 or len(cached)<int(args.min_audio_seconds*16000) or not np.isfinite(cached).all() or float(np.max(np.abs(cached)))<1e-7:
                    raise ValueError('invalid/silent existing .npy cache')
                del cached
            except Exception as exc:
                failed.append(dict(recording_id=r.recording_id,file_path=r.file_path,error=str(exc)[:500]))
                log('[CACHE ERROR]',r.recording_id,exc)
                continue
            if r.dataset=='PME4' and args.pme4_speech_trim:
                try: st,ed=float(r.speech_start),float(r.speech_end)
                except (TypeError,ValueError): st,ed=float('nan'),float('nan')
                trimmed=bool(np.isfinite(st) and np.isfinite(ed) and 0<=st<ed<=float(r.duration_sec)+.05 and ed-st>=.12)
        if trimmed:used_trim+=1
        out.append(dict(recording_id=r.recording_id,cache_path=str(path),trim_applied=bool(trimmed),
                        processed_samples=int(np.load(path,mmap_mode='r').shape[0])))
        if j%500==0:log('[CACHE]',j,'/',len(good),'audio files processed')
    pd.DataFrame(failed,columns=['recording_id','file_path','error']).to_csv(project/'audit'/'preprocessing_errors.csv',index=False)
    if failed:raise RuntimeError(f'{len(failed)} preprocessing failures. Review audit/preprocessing_errors.csv; '
                                'repair data or use a NEW project root. All six folds must use identical fixed data.')
    mapping=pd.DataFrame(out).sort_values('recording_id').reset_index(drop=True)
    mapping.to_csv(project/'cache'/'audio_cache_manifest.csv',index=False)
    atomic_json(project/'cache'/'audio_cache_summary.json',dict(n_cached=len(mapping),trimmed_pme4=used_trim,
                fallback_pme4_metadata=skipped_trim,settings=settings))
    log('[PREPARED] cache',len(mapping),'clips; PME4 trims',used_trim,'fallback trims',skipped_trim)
    return mapping


def cache_manifest(args,project,audit):
    path=project/'cache'/'audio_cache_manifest.csv'
    if not path.is_file():raise RuntimeError('Missing audio cache manifest. Run --mode prepare first.')
    settings=cache_settings(args)
    meta=project/'cache'/'cache_config.json'
    if not meta.is_file() or json.loads(meta.read_text())!=settings:
        raise RuntimeError('Cache settings mismatch. Preserve preprocessing for all six folds.')
    mapping=pd.read_csv(path)
    good=audit[audit.status=='usable']
    if set(mapping.recording_id)!=set(good.recording_id):raise RuntimeError('Audio cache manifest does not match audit')
    if not all(Path(p).is_file() for p in mapping.cache_path):raise RuntimeError('Missing audio cache entries. Run --mode prepare.')
    # Detect changes to source files after locking the paper protocol.
    for r in good.itertuples(index=False):
        st=Path(r.file_path).stat()
        if int(st.st_size)!=int(r.file_bytes) or int(st.st_mtime_ns)!=int(r.file_mtime_ns):
            raise RuntimeError(f'Audio file changed since audit: {r.file_path}. Start a new experiment root.')
    return mapping


class AudioCropDataset:
    """One deterministic random crop/trial per train epoch; fixed windows for eval."""
    def __init__(self,records,mode,crop_sec=4.0,eval_windows=5,seed=2026,epoch=0):
        self.records=records.reset_index(drop=True)
        self.mode=mode
        self.nsamples=int(round(crop_sec*16000))
        self.seed=seed
        self.epoch=epoch
        self.items=[]
        if mode=='train':
            self.items=[(i,None) for i in range(len(self.records))]
        else:
            for i,r in enumerate(self.records.itertuples(index=False)):
                length=int(r.processed_samples)
                starts=eval_positions(length,self.nsamples,eval_windows)
                for st in starts:self.items.append((i,st))
        # Bound per-worker decoded waveform RAM; never accumulate thousands of open mmaps.
        self.open_files=OrderedDict()

    def __len__(self):return len(self.items)

    def __getitem__(self,idx):
        i,start=self.items[idx]
        r=self.records.iloc[i]
        if i not in self.open_files:
            self.open_files[i]=np.load(r.cache_path,allow_pickle=False)
            if len(self.open_files)>32:self.open_files.popitem(last=False)
        else:self.open_files.move_to_end(i)
        wav=self.open_files[i];n=len(wav)
        if start is None:
            rng=np.random.default_rng(self.seed + self.epoch*1_000_003 + (i+1)*1097)
            start=int(rng.integers(0,max(0,n-self.nsamples)+1))
        segment=np.asarray(wav[start:start+self.nsamples],dtype=np.float32).copy()
        valid=len(segment)
        if valid<self.nsamples:segment=np.pad(segment,(0,self.nsamples-valid))
        if valid:
            x=segment[:valid]
            std=float(np.sqrt(np.mean((x-x.mean())**2)))
            segment[:valid]=np.clip((x-x.mean())/max(std,1e-5),-8.0,8.0)
        segment[valid:]=0.0
        return torch.from_numpy(segment),int(valid),int(r.class_id),int(i)


def eval_positions(n,window,max_windows):
    if n<=window:return [0]
    max_st=n-window
    # Each longer recording contributes equal max_windows, not more weight.
    return sorted(set(int(round(x)) for x in np.linspace(0,max_st,max_windows)))


def collate(xs):
    a,lengths,labels,index=zip(*xs)
    # Shape [B, 4*16000]. Pad is already zero, attention mask marks real samples.
    values=torch.stack(a)
    m=torch.arange(values.shape[1])[None,:] < torch.tensor(lengths)[:,None]
    return values,m.long(),torch.tensor(labels,dtype=torch.long),torch.tensor(index,dtype=torch.long)


class CorpusEmotionSampler:
    """Balanced by both source corpus and emotion. No target-corpus access."""
    def __init__(self,records,n_samples,seed=2026):
        self.records=records.reset_index(drop=True)
        self.n_samples=int(n_samples)
        self.seed=seed
        self.epoch=0
        groups=defaultdict(list)
        for i,r in enumerate(self.records.itertuples(index=False)):
            groups[(r.dataset,int(r.class_id))].append(i)
        corpora=set(self.records.dataset)
        if len(corpora)!=5:raise ValueError('Sampler requires exactly 5 source corpora')
        if len(groups)!=5*4:raise ValueError('All source corpus/emotion cells must be nonempty')
        self.keys=sorted(groups)
        self.groups={k:np.asarray(v,dtype=np.int64) for k,v in groups.items()}

    def __len__(self):return self.n_samples

    def __iter__(self):
        rng=np.random.default_rng(self.seed+1_000_003*self.epoch)
        order=rng.permutation(self.n_samples)
        result=np.empty(self.n_samples,dtype=np.int64)
        for j,k in enumerate(self.keys):
            places=order[j::len(self.keys)]
            pool=self.groups[k]
            result[places]=rng.choice(pool,size=len(places),replace=True)
        return iter(result.tolist())


class WavLMBaseline(nn.Module):
    """Exactly one pretrained encoder, masked mean temporal pooling, linear head."""
    def __init__(self,backbone,freeze_top=6,dropout=.2):
        super().__init__()
        self.wavlm=backbone
        self.norm=nn.LayerNorm(backbone.config.hidden_size)
        self.dropout=nn.Dropout(dropout)
        self.classifier=nn.Linear(backbone.config.hidden_size,4)
        self.finetune_top=freeze_top
        self.stage='frozen'
        self.set_stage('frozen')

    def set_stage(self,stage):
        assert stage in ('frozen','finetune')
        self.stage=stage
        self.wavlm.requires_grad_(False)
        if stage=='finetune':
            for layer in self.wavlm.encoder.layers[-self.finetune_top:]:layer.requires_grad_(True)

    def train(self,mode=True):
        super().train(mode)
        if mode and self.stage=='frozen':self.wavlm.eval()
        return self

    def forward(self,x,mask):
        if self.stage=='frozen':
            with torch.no_grad():h=self.wavlm(input_values=x,attention_mask=mask,return_dict=True).last_hidden_state
        else:h=self.wavlm(input_values=x,attention_mask=mask,return_dict=True).last_hidden_state
        nvalid=mask.long().sum(1)
        # WavLM's own exact convolutional frame-length computation.
        lengths=self.wavlm._get_feat_extract_output_lengths(nvalid).to(h.device).clamp(1,h.shape[1])
        m=(torch.arange(h.shape[1],device=h.device)[None,:]<lengths[:,None]).to(h.dtype)
        pooled=((h*m.unsqueeze(-1)).float().sum(1)/m.float().sum(1,keepdim=True).clamp(min=1))
        pooled=self.norm(pooled)
        logits=self.classifier(self.dropout(pooled))
        return logits,pooled


def prepare_transformers_speech_only():
    """Disable optional TorchVision *inside this process only* before HF imports.

    Some mixed CUDA images have TorchVision installed but missing its nms op.
    Transformers treats TorchVision as optional; WavLM does not use it. On
    affected installations, the optional image import raises during WavLM
    class loading. This patch changes no installed packages or CUDA libraries.

    Must run before `transformers.image_utils` or `transformers.video_utils`
    are imported. Use a clean speech-only Python invocation, not a notebook
    process that already loaded Transformers vision helpers.
    """
    import sys
    import transformers
    from transformers.utils import import_utils

    available = getattr(import_utils, '_torchvision_available', None)
    if available is None:
        # A future version may change its internal optional-dependency API.
        # Fail explicitly rather than risk loading random model substitutes.
        raise RuntimeError('Unsupported Transformers optional TorchVision API: '
                           f'version={transformers.__version__}. Install a '
                           'compatible speech-only Transformers release.')
    if not isinstance(available, bool):
        raise RuntimeError('Unexpected Transformers TorchVision flag type: '
                           f'{type(available).__name__}')

    if available:
        loaded_vision = [name for name in (
            'transformers.image_utils', 'transformers.image_transforms',
            'transformers.video_utils') if name in sys.modules]
        if loaded_vision:
            raise RuntimeError('Transformers vision modules were imported before '
                               f'the speech-only compatibility hook: {loaded_vision}. '
                               'Restart in a fresh Python process.')
        import_utils._torchvision_available = False
        print('[SPEECH IMPORT FIX] Disabled optional TorchVision availability '
              'within this Python process only; installed libraries untouched.',
              flush=True)
    if import_utils.is_torchvision_available():
        raise RuntimeError('TorchVision optional-dependency bypass did not apply')


def get_pretrained_model(args):
    prepare_transformers_speech_only()
    from transformers import WavLMConfig, WavLMModel
    log('[PRETRAINED]',MODEL_ID,'commit',MODEL_REVISION)
    config=WavLMConfig.from_pretrained(MODEL_ID,revision=MODEL_REVISION)
    if config.hidden_size!=1024 or config.num_hidden_layers!=24:
        raise RuntimeError('Expected WavLM-Large (24 layers, 1024 hidden size)')
    config.apply_spec_augment=False
    config.layerdrop=0.0
    encoder=WavLMModel.from_pretrained(MODEL_ID,revision=MODEL_REVISION,config=config)
    try:encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    except TypeError:encoder.gradient_checkpointing_enable()
    return WavLMBaseline(encoder,freeze_top=args.finetune_layers,dropout=args.dropout)


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def amp_ctx(device):
    return torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()


def make_loader(data,args,mode,epoch=0):
    ds=AudioCropDataset(data,mode,crop_sec=args.crop_seconds,eval_windows=args.eval_windows,
                        seed=args.seed,epoch=epoch)
    if mode=='train':
        sampler=CorpusEmotionSampler(data,len(data),seed=args.seed+args.fold_seed)
        sampler.epoch=epoch
        return torch.utils.data.DataLoader(ds,batch_size=args.microbatch,sampler=sampler,
                num_workers=args.workers,pin_memory=torch.cuda.is_available(),collate_fn=collate,
                persistent_workers=args.workers>0)
    return torch.utils.data.DataLoader(ds,batch_size=args.eval_batch,shuffle=False,
            num_workers=args.workers,pin_memory=torch.cuda.is_available(),collate_fn=collate,
            persistent_workers=args.workers>0)


@torch.no_grad()
def predict_records(model,records,args,device):
    from tqdm.auto import tqdm
    if len(records)==0:raise ValueError('No records to evaluate')
    loader=make_loader(records,args,'eval')
    model.eval()
    n=len(loader.dataset.records)
    sums=np.zeros((n,4),np.float64)
    embs=np.zeros((n,1024),np.float64)
    nums=np.zeros(n,np.int32)
    for x,mask,_,ix in tqdm(loader,desc='Recording eval',leave=False,dynamic_ncols=True):
        x=x.to(device,non_blocking=True);mask=mask.to(device,non_blocking=True)
        with amp_ctx(device):logits,features=model(x,mask)
        k=ix.numpy()
        np.add.at(sums,k,logits.float().cpu().numpy())
        np.add.at(embs,k,features.float().cpu().numpy())
        np.add.at(nums,k,1)
    if (nums==0).any():raise RuntimeError('Missing prediction for some recordings')
    scores=sums/nums[:,None]
    scores-=scores.max(1,keepdims=True)
    scores=np.exp(scores);scores/=scores.sum(1,keepdims=True)
    return scores.astype('float32'),(embs/nums[:,None]).astype('float32'),nums


def metric_summary(data,probabilities):
    from sklearn.metrics import accuracy_score,f1_score,precision_recall_fscore_support,confusion_matrix
    y=data.class_id.to_numpy(dtype=int)
    yp=probabilities.argmax(-1)
    by_sub=[];by_class=[]
    for subject,part in data.groupby('speaker_id',sort=True):
        ix=part.index.to_numpy()
        yt=y[ix];pred=yp[ix]
        p,r,f,s=precision_recall_fscore_support(yt,pred,labels=range(4),zero_division=0)
        by_sub.append(dict(dataset=part.dataset.iloc[0],speaker_id=subject,n_records=len(ix),
                           accuracy=float(accuracy_score(yt,pred)),
                           macro_f1=float(f1_score(yt,pred,labels=range(4),average='macro',zero_division=0))))
        for c in range(4):
            by_class.append(dict(dataset=part.dataset.iloc[0],speaker_id=subject,
                                 emotion=CLASS_NAMES[c],precision=float(p[c]),recall=float(r[c]),
                                 f1=float(f[c]),support=int(s[c])))
    per_subject=pd.DataFrame(by_sub)
    sc=pd.DataFrame(by_class)
    p,r,f,s=precision_recall_fscore_support(y,yp,labels=range(4),zero_division=0)
    classes=pd.DataFrame([dict(emotion=CLASS_NAMES[i],precision=float(p[i]),recall=float(r[i]),
                               f1=float(f[i]),support=int(s[i])) for i in range(4)])
    metrics=dict(n_recordings=len(y),n_subjects=len(per_subject),
                 pooled_accuracy=float(accuracy_score(y,yp)),
                 macro_f1=float(f1_score(y,yp,labels=range(4),average='macro',zero_division=0)),
                 uar=float(np.mean(r)),mean_subject_accuracy=float(per_subject.accuracy.mean()),
                 mean_subject_macro_f1=float(per_subject.macro_f1.mean()),
                 std_subject_accuracy=float(per_subject.accuracy.std(ddof=1)) if len(per_subject)>1 else 0.,
                 confusion_matrix=confusion_matrix(y,yp,labels=range(4)).tolist())
    rng=np.random.default_rng(20261108)
    boot=rng.choice(per_subject.accuracy.to_numpy(),size=(2000,len(per_subject)),replace=True).mean(axis=1)
    metrics['mean_subject_accuracy_95ci_bootstrap']=[float(v) for v in np.percentile(boot,[2.5,97.5])]
    domains=[]
    for corpus,group in data.groupby('dataset',sort=False):
        ids=group.index.to_numpy();yy=y[ids];pp=yp[ids]
        sub=per_subject[per_subject.dataset==corpus]
        domains.append(dict(dataset=corpus,n_recordings=len(ids),n_subjects=len(sub),
                            pooled_accuracy=float(accuracy_score(yy,pp)),
                            macro_f1=float(f1_score(yy,pp,labels=range(4),average='macro',zero_division=0)),
                            mean_subject_accuracy=float(sub.accuracy.mean()),
                            uar=float(precision_recall_fscore_support(yy,pp,labels=range(4),zero_division=0)[1].mean())))
    return metrics,per_subject,classes,sc,pd.DataFrame(domains),yp


def export_report(dest,data,probs,embs,nums,title):
    from sklearn.metrics import ConfusionMatrixDisplay,confusion_matrix,roc_curve,auc
    dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
    data=data.reset_index(drop=True)
    metrics,subjects,classes,subclass,corpora,pred=metric_summary(data,probs)
    out=data[['recording_id','dataset','speaker_id','emotion','class_id','file_path','duration_sec']].copy()
    out['prediction_id']=pred
    out['predicted_emotion']=[CLASS_NAMES[x] for x in pred]
    out['correct']=(out.class_id==out.prediction_id).astype(int)
    out['n_eval_windows']=nums
    for i,k in enumerate(CLASS_NAMES):out['prob_'+k.lower()]=probs[:,i]
    out.to_csv(dest/'recording_predictions.csv',index=False)
    subjects.to_csv(dest/'per_subject_metrics.csv',index=False)
    classes.to_csv(dest/'per_class_metrics.csv',index=False)
    subclass.to_csv(dest/'per_subject_class_metrics.csv',index=False)
    corpora.to_csv(dest/'per_corpus_metrics.csv',index=False)
    np.savez_compressed(dest/'recording_embeddings_1024d.npz',
                        recording_id=data.recording_id.to_numpy(dtype='U160'),
                        dataset=data.dataset.to_numpy(dtype='U16'),
                        speaker_id=data.speaker_id.to_numpy(dtype='U32'),
                        emotion_id=data.class_id.to_numpy(dtype=np.int16),
                        embedding=embs.astype('float16'),probabilities=probs.astype('float32'))
    metrics['report_name']=title
    metrics['class_order']=list(CLASS_NAMES)
    metrics['head']='LayerNorm + dropout + Linear(1024,4)'
    atomic_json(dest/'metrics.json',metrics)
    if len(subjects):
        sorted_sub=subjects.sort_values('accuracy',ascending=True)
        fig,ax=plt.subplots(figsize=(12,max(5, min(22,len(sorted_sub)*.22))))
        ax.barh(np.arange(len(sorted_sub)),sorted_sub.accuracy*100,color='#537EAE')
        ax.set_yticks(np.arange(len(sorted_sub)),[x.split(':')[-1] for x in sorted_sub.speaker_id],fontsize=7)
        ax.set(xlim=(0,100),xlabel='Recording accuracy (%)',title=title+' — per-subject accuracy')
        ax.axvline(metrics['mean_subject_accuracy']*100,color='#A9463E',linestyle='--',label=f'Mean subject acc. = {100*metrics["mean_subject_accuracy"]:.2f}%')
        ax.legend(loc='lower right');fig.tight_layout();fig.savefig(dest/'per_subject_accuracy.png',dpi=160);plt.close(fig)
        mat=subclass.pivot(index='speaker_id',columns='emotion',values='recall').reindex(columns=CLASS_NAMES)
        fig,ax=plt.subplots(figsize=(7,max(5,min(22,len(mat)*.22))))
        im=ax.imshow(mat.to_numpy(),vmin=0,vmax=1,cmap='Blues',aspect='auto')
        ax.set_xticks(range(4),CLASS_NAMES);ax.set_yticks(range(len(mat)),mat.index,fontsize=7)
        ax.set_title(title+' — recall by speaker and class');fig.colorbar(im,ax=ax,label='Recall')
        fig.tight_layout();fig.savefig(dest/'subject_class_recall_heatmap.png',dpi=160);plt.close(fig)
    for normalize,stem in ((None,'counts'),('true','row_normalized')):
        cm=confusion_matrix(data.class_id,pred,labels=list(range(4)),normalize=normalize)
        fig,ax=plt.subplots(figsize=(7,6))
        ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(ax=ax,cmap='Blues',colorbar=False,
                        values_format='.2f' if normalize else 'd')
        ax.set_title(title+' — confusion '+stem)
        fig.tight_layout();fig.savefig(dest/f'confusion_{stem}.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4.5));n=np.arange(4);w=.25
    for j,col in enumerate(['precision','recall','f1']):
        ax.bar(n+(j-1)*w,classes[col],w,label=col.title())
    ax.set(ylim=(0,1.03),ylabel='Score',title=title+' — per-class metrics')
    ax.set_xticks(n,CLASS_NAMES);ax.legend();fig.tight_layout()
    fig.savefig(dest/'class_precision_recall_f1.png',dpi=180);plt.close(fig)
    from sklearn.preprocessing import label_binarize
    yt=label_binarize(data.class_id.to_numpy(),classes=[0,1,2,3])
    fig,ax=plt.subplots(figsize=(7,5))
    for i,k in enumerate(CLASS_NAMES):
        if yt[:,i].sum()==0 or yt[:,i].sum()==len(yt):continue
        fpr,tpr,_=roc_curve(yt[:,i],probs[:,i]);ax.plot(fpr,tpr,label=f'{k} (AUC {auc(fpr,tpr):.3f})')
    ax.plot([0,1],[0,1],'--',color='#666',linewidth=1)
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title=title+' — One-vs-rest ROC')
    ax.legend();fig.tight_layout();fig.savefig(dest/'roc_ovr.png',dpi=160);plt.close(fig)
    per_dir=dest/'per_subject_confusion_matrices';per_dir.mkdir(exist_ok=True)
    for speaker,part in out.groupby('speaker_id'):
        cm=confusion_matrix(part.class_id,part.prediction_id,labels=range(4))
        fig,ax=plt.subplots(figsize=(5,4));ConfusionMatrixDisplay(cm,display_labels=CLASS_NAMES).plot(
                    ax=ax,cmap='Blues',colorbar=False,values_format='d')
        ax.set_title(f'{title} — {speaker}');fig.tight_layout()
        fig.savefig(per_dir/(speaker.replace(':','_')+'.png'),dpi=120);plt.close(fig)
    return metrics,corpora


def write_training_plots(path,hist,target):
    if not hist:return
    d=pd.DataFrame(hist);d.to_csv(path/'training_history.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(d.epoch,d.train_loss,marker='o',label='train CE')
    ax.set(xlabel='Epoch',ylabel='Training cross-entropy',title=f'{target}: training loss');ax.legend()
    fig.tight_layout();fig.savefig(path/'training_loss.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(d.epoch,d.val_mean_corpus_macro_f1,marker='o',label='Source validation: mean 5-corpus macro-F1')
    ax.plot(d.epoch,d.val_mean_corpus_msa,marker='o',label='Source validation: mean 5-corpus MSA')
    ax.set(xlabel='Epoch',ylabel='Score',ylim=(0,1),title=f'{target}: source validation')
    ax.legend();fig.tight_layout();fig.savefig(path/'validation_curves.png',dpi=180);plt.close(fig)
    vals=[]
    for c in CORPORA:
        if c==target:continue
        key='val_'+FOLD_NAMES[c]+'_macro_f1'
        if key in d.columns:vals.append((c,key))
    fig,ax=plt.subplots(figsize=(10,4.5))
    for c,key in vals:ax.plot(d.epoch,d[key],marker='.',label=c)
    ax.set(xlabel='Epoch',ylabel='Macro-F1',ylim=(0,1),title=f'{target}: individual source-corpus validation')
    ax.legend(ncol=3);fig.tight_layout();fig.savefig(path/'per_source_validation_curves.png',dpi=180);plt.close(fig)


def make_optimizer(model,args,stage,updates):
    if stage=='frozen':
        params=[x for x in model.parameters() if x.requires_grad]
        groups=[{'params':params,'lr':args.head_lr_frozen}]
    else:
        groups=[{'params':list(model.norm.parameters())+list(model.classifier.parameters()),'lr':args.head_lr}]
        layers=model.wavlm.encoder.layers[-args.finetune_layers:]
        for pos,layer in enumerate(layers):
            lr=args.backbone_lr*(args.layer_decay**(len(layers)-pos-1))
            groups.append({'params':[p for p in layer.parameters() if p.requires_grad],'lr':lr})
    optimizer=torch.optim.AdamW(groups,weight_decay=args.weight_decay)
    total=max(1,updates)
    warmup=max(1,int(total*args.warmup_frac))
    def factor(step):
        if step<warmup:return .05+.95*(step+1)/warmup
        t=min(1.,(step-warmup)/max(1,total-warmup))
        return .05+.95*.5*(1+math.cos(math.pi*t))
    scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=factor)
    return optimizer,scheduler


def save_pth(path,payload):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.writing')
    torch.save(payload,temp)
    os.replace(temp,path)


def model_meta(args,audit_sha,fold_target,epoch,best_score,best_epoch):
    return dict(model=MODEL_ID,revision=MODEL_REVISION,architecture='WavLM-Large + masked-mean + LayerNorm + Dropout + Linear(1024,4)',
                class_order=list(CLASS_NAMES),target=fold_target,audit_sha256=audit_sha,
                epoch=epoch,best_source_validation_mean_macro_f1=best_score,best_epoch=best_epoch,
                config={'seed':args.seed,'crop_seconds':args.crop_seconds,'eval_windows':args.eval_windows,
                        'pme4_speech_trim':args.pme4_speech_trim,'finetune_layers':args.finetune_layers,
                        'max_epochs':args.epochs,'freeze_epochs':args.freeze_epochs,
                        'microbatch':args.microbatch,'grad_accum':args.grad_accum})


def strict_load(model,checkpoint,args,audit_sha,target):
    obj=torch.load(checkpoint,map_location='cpu',weights_only=False)
    meta=obj['meta']
    if meta['target']!=target or meta['audit_sha256']!=audit_sha or meta['class_order']!=list(CLASS_NAMES):
        raise RuntimeError('Checkpoint belongs to a different target, audit, or label order')
    expected={'seed':args.seed,'crop_seconds':args.crop_seconds,'eval_windows':args.eval_windows,
              'pme4_speech_trim':args.pme4_speech_trim,'finetune_layers':args.finetune_layers,
              'max_epochs':args.epochs,'freeze_epochs':args.freeze_epochs,
              'microbatch':args.microbatch,'grad_accum':args.grad_accum}
    if meta['config'] != expected:
        raise RuntimeError('Training/evaluation settings have changed since checkpoint creation: '+str(meta['config']))
    model.load_state_dict(obj['model'],strict=True)
    return obj


def epoch_train(model,train,args,opt,sched,device,epoch):
    from tqdm.auto import tqdm
    model.train()
    loader=make_loader(train,args,'train',epoch)
    opt.zero_grad(set_to_none=True)
    total_loss=0.;correct=0;count=0
    steps=0
    for i,(x,mask,y,_) in enumerate(tqdm(loader,desc=f'epoch {epoch} train',dynamic_ncols=True,leave=False)):
        x=x.to(device,non_blocking=True);mask=mask.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with amp_ctx(device):
            logits,_=model(x,mask)
            loss=nn.functional.cross_entropy(logits,y)
        (loss/args.grad_accum).backward()
        total_loss+=float(loss.item())*len(y)
        correct+=int((logits.argmax(-1)==y).sum().item())
        count+=len(y)
        if (i+1)%args.grad_accum==0 or (i+1)==len(loader):
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0)
            opt.step();sched.step();opt.zero_grad(set_to_none=True)
            steps+=1
        if (i+1)%max(1,args.progress_batches)==0:
            log('[TRAIN]',f'epoch={epoch}',f'batches={i+1}/{len(loader)}',f'loss={total_loss/max(count,1):.4f}',
                f'crop_acc={100*correct/max(count,1):.2f}%',f'updates={steps}')
    return dict(train_loss=total_loss/max(count,1),train_crop_accuracy=correct/max(count,1),updates=steps)


def train_fold(args,project,audit,speakers,cache,audit_sha,target):
    folder=project/'folds'/FOLD_NAMES[target]
    folder.mkdir(parents=True,exist_ok=True)
    manifest=make_fold_manifest(audit,speakers,target)
    manifest=manifest.merge(cache,on='recording_id',how='inner',validate='one_to_one')
    if len(manifest)!=(audit.status=='usable').sum():raise RuntimeError('Cache merge dropped recordings')
    manifest.to_csv(folder/'immutable_split_manifest.csv',index=False)
    train=manifest[manifest.split=='train'].reset_index(drop=True)
    val=manifest[manifest.split=='val'].reset_index(drop=True)
    test=manifest[manifest.split=='test'].reset_index(drop=True)
    assert test.dataset.nunique()==1 and test.dataset.iloc[0]==target
    done=folder/'training_complete.json'
    if done.is_file():
        log('[SKIP]',target,'already frozen; best model is',folder/'best.pth')
        return
    seed=args.seed + 1009*CORPORA.index(target)
    set_seed(seed);args.fold_seed=seed
    device=torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    if args.device=='cuda' and device.type!='cuda':raise RuntimeError('CUDA requested but unavailable')
    log('[FOLD START]',target, 'train',len(train),'val',len(val),'sealed test',len(test),
        'train speakers',train.speaker_id.nunique(),'val speakers',val.speaker_id.nunique())
    model=get_pretrained_model(args).to(device)
    best_path=folder/'best.pth';last_path=folder/'last.pth'
    history=[];best_score=-1.0;best_epoch=0;stalled=0;start_epoch=1
    previous=None
    if last_path.is_file() and not args.resume:
        raise RuntimeError(f'Existing interrupted checkpoint at {last_path}. Use --resume to continue, or a new project root.')
    if args.resume and last_path.is_file():
        previous=strict_load(model,last_path,args,audit_sha,target)
        history=previous['history'];best_score=float(previous['best_score'])
        best_epoch=int(previous['best_epoch']);stalled=int(previous['stalled'])
        start_epoch=int(previous['meta']['epoch'])+1
        if 'rng_torch' in previous:
            torch.set_rng_state(previous['rng_torch'])
            np.random.set_state(previous['rng_numpy'])
            random.setstate(previous['rng_python'])
            if device.type=='cuda' and 'rng_cuda' in previous:torch.cuda.set_rng_state_all(previous['rng_cuda'])
        log('[RESUME]',target,'epoch',start_epoch,'best',best_score)
    cur_stage=None;optimizer=None;scheduler=None
    batch_steps=math.ceil(len(train)/args.microbatch)
    updates_per_epoch=math.ceil(batch_steps/args.grad_accum)
    for epoch in range(start_epoch,args.epochs+1):
        stage='frozen' if epoch<=args.freeze_epochs else 'finetune'
        if stage!=cur_stage:
            model.set_stage(stage)
            stage_n=args.freeze_epochs if stage=='frozen' else args.epochs-args.freeze_epochs
            optimizer,scheduler=make_optimizer(model,args,stage,max(1,stage_n)*updates_per_epoch)
            cur_stage=stage
            if previous is not None and int(previous['meta']['epoch'])+1==epoch and previous['stage']==stage:
                optimizer.load_state_dict(previous['optimizer'])
                scheduler.load_state_dict(previous['scheduler'])
                log('[RESUME]',target,'restored optimizer state',stage)
            previous=None
            log('[STAGE]',target,stage,'trainable params',sum(p.numel() for p in model.parameters() if p.requires_grad))
        started=time.time()
        losses=epoch_train(model,train,args,optimizer,scheduler,device,epoch)
        probs,_,_=predict_records(model,val,args,device)
        val_metrics,sub,classes,subcls,domains,_=metric_summary(val,probs)
        if set(domains.dataset)!=set(CORPORA)-{target}:
            raise AssertionError('Missing source corpus in validation')
        score=float(domains.macro_f1.mean())
        val_msa=float(domains.mean_subject_accuracy.mean())
        entry={'epoch':epoch,'stage':stage,**losses,
               'val_mean_corpus_macro_f1':score,'val_mean_corpus_msa':val_msa,
               'val_pooled_macro_f1':val_metrics['macro_f1'],
               'val_all_source_mean_subject_accuracy':val_metrics['mean_subject_accuracy'],
               'epoch_minutes':(time.time()-started)/60}
        entry.update({'val_'+FOLD_NAMES[r.dataset]+'_macro_f1':float(r.macro_f1) for r in domains.itertuples()})
        history.append(entry)
        improvement=score>best_score+args.min_delta
        if improvement:
            best_score=score;best_epoch=epoch;stalled=0
            save_pth(best_path,dict(meta=model_meta(args,audit_sha,target,epoch,best_score,best_epoch),
                                    model=model.state_dict()))
            log('[BEST]',target,'epoch',epoch,'mean source-val corpus F1',round(score,5))
        elif stage=='finetune':stalled+=1
        save_pth(last_path,dict(meta=model_meta(args,audit_sha,target,epoch,best_score,best_epoch),
                    model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
                    stage=stage,stalled=stalled,best_score=best_score,best_epoch=best_epoch,
                    history=history,rng_torch=torch.get_rng_state(),rng_numpy=np.random.get_state(),
                    rng_python=random.getstate(),
                    **({'rng_cuda':torch.cuda.get_rng_state_all()} if device.type=='cuda' else {})))
        write_training_plots(folder,history,target)
        log('[EPOCH DONE]',target,epoch,'/',args.epochs,'source macro-F1',f'{score:.4f}',
            'mean corpus MSA',f'{val_msa:.4f}','best',best_epoch,'patience',f'{stalled}/{args.patience}')
        if stage=='finetune' and stalled>=args.patience:
            log('[EARLY STOP]',target,'epoch',epoch)
            break
    if not best_path.is_file():raise RuntimeError('No best checkpoint saved')
    # source validation report using frozen, selected best checkpoint only
    obj=strict_load(model,best_path,args,audit_sha,target)
    model.eval()
    vp,ve,vn=predict_records(model,val,args,device)
    vm,vc=export_report(folder/'source_validation',val,vp,ve,vn,f'{target}: five-source validation')
    atomic_json(done,dict(target=target,best_epoch=best_epoch,best_source_val_mean_corpus_macro_f1=best_score,
                          epochs_run=len(history),source_validation_metrics=vm,
                          source_val_unweighted_corpus_macro_f1=float(vc.macro_f1.mean()),
                          train_samples=len(train),val_samples=len(val),sealed_target_test_samples=len(test),
                          train_corpora=[x for x in CORPORA if x!=target],
                          target_corpus_not_used_in_training_or_validation=True,
                          audit_sha256=audit_sha))
    log('[FROZEN]',target,'best epoch',best_epoch,'best source macro-F1',best_score)
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def test_fold(args,project,audit,speakers,cache,audit_sha,target):
    folder=project/'folds'/FOLD_NAMES[target]
    done=folder/'training_complete.json'
    if not done.is_file():raise RuntimeError(f'Fold {target} training has not completed. Test is locked.')
    result=folder/'external_test'/'metrics.json'
    if result.is_file() and not args.force_retest:
        log('[TEST ALREADY COMPLETE]',target,'stored',result);return
    if not args.unlock_test:raise RuntimeError('External corpus testing requires --unlock-test')
    set_seed(args.seed+1009*CORPORA.index(target));args.fold_seed=args.seed+1009*CORPORA.index(target)
    manifest=make_fold_manifest(audit,speakers,target)
    manifest=manifest.merge(cache,on='recording_id',how='inner',validate='one_to_one')
    test=manifest[manifest.split=='test'].reset_index(drop=True)
    if set(test.dataset)!={target}:raise AssertionError('Test includes non-target corpus')
    device=torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    log('[TEST]',target,'all unseen-corpus recordings',len(test),'subjects',test.speaker_id.nunique())
    model=get_pretrained_model(args).to(device)
    strict_load(model,folder/'best.pth',args,audit_sha,target)
    model.eval()
    probs,embs,counts=predict_records(model,test,args,device)
    metrics,_=export_report(folder/'external_test',test,probs,embs,counts,f'{target}: external zero-shot test')
    # separate metadata for research audit
    metadata=dict(target=target,n_target_recordings=len(test),n_target_speakers=test.speaker_id.nunique(),
                  model=MODEL_ID,checkpoint=str(folder/'best.pth'),
                  no_target_training_or_selection=True,checkpoint_chosen_by='five SOURCE corpus validation',
                  measured_once=True,metrics=metrics)
    atomic_json(folder/'external_test'/'external_test_provenance.json',metadata)
    log('[EXTERNAL RESULT]',target,'MSA',f'{metrics["mean_subject_accuracy"]:.4%}',
        'macro-F1',f'{metrics["macro_f1"]:.4%}','UAR',f'{metrics["uar"]:.4%}')
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def aggregate(project):
    rows=[]
    for target in CORPORA:
        folder=project/'folds'/FOLD_NAMES[target]
        path=folder/'external_test'/'metrics.json'
        if not path.exists():
            log('[AGGREGATE] Missing result:',target)
            continue
        m=json.loads(path.read_text())
        val=json.loads((folder/'training_complete.json').read_text())
        rows.append(dict(target_corpus=target,train_corpora=' + '.join([x for x in CORPORA if x!=target]),
            train_recordings=val['train_samples'],source_val_recordings=val['val_samples'],
            target_test_recordings=m['n_recordings'],test_speakers=m['n_subjects'],
            selected_epoch=val['best_epoch'],source_val_mean_corpus_macro_f1=val['best_source_val_mean_corpus_macro_f1'],
            test_mean_subject_accuracy=m['mean_subject_accuracy'],test_pooled_accuracy=m['pooled_accuracy'],
            test_macro_f1=m['macro_f1'],test_uar=m['uar'],
            test_mean_subject_macro_f1=m['mean_subject_macro_f1'],
            msa_ci_low=m['mean_subject_accuracy_95ci_bootstrap'][0],
            msa_ci_high=m['mean_subject_accuracy_95ci_bootstrap'][1]))
    if not rows:raise RuntimeError('No completed external tests. Run --mode test --unlock-test.')
    dest=project/'paper_ready';dest.mkdir(parents=True,exist_ok=True)
    df=pd.DataFrame(rows)
    df.to_csv(dest/'six_fold_loco_results.csv',index=False)
    mean_macro=float(df.test_macro_f1.mean())
    mean_uar=float(df.test_uar.mean())
    mean_msa=float(df.test_mean_subject_accuracy.mean())
    rng=np.random.default_rng(20261108)
    ci={}
    for field in ('test_macro_f1','test_uar','test_mean_subject_accuracy'):
        bootstrap=rng.choice(df[field].to_numpy(),size=(5000,len(df)),replace=True).mean(axis=1)
        ci[field+'_corpus_bootstrap_95ci']=[float(x) for x in np.percentile(bootstrap,[2.5,97.5])]
    totals=dict(n_completed=len(df),n_expected=6,primary_metric='unweighted mean test macro-F1 across six corpora',
                mean_test_macro_f1=mean_macro,mean_test_uar=mean_uar,
                mean_test_subject_accuracy=mean_msa,
                test_samples_total=int(df.target_test_recordings.sum()),
                descriptive_cross_corpus_uncertainty=ci,
                caution='Each of six target corpora is fully unseen in its own fold; speakers within source corpora are disjoint between source train and validation.')
    atomic_json(dest/'loco_summary.json',totals)
    pretty=df[['target_corpus','test_speakers','target_test_recordings','test_mean_subject_accuracy','test_macro_f1','test_uar','test_pooled_accuracy']].copy()
    for col in ['test_mean_subject_accuracy','test_macro_f1','test_uar','test_pooled_accuracy']:
        pretty[col]=(pretty[col]*100).round(2)
    pretty.columns=['Held-out corpus','Speakers','Recordings','MSA (%)','Macro-F1 (%)','UAR (%)','Pooled Acc. (%)']
    md='# Leave-one-corpus-out evaluation — plain WavLM-Large\n\n'
    md+='| '+ ' | '.join(pretty.columns) +' |\n|'+ '|'.join(['---']*len(pretty.columns))+'|\n'
    for _,r in pretty.iterrows():md+='| '+' | '.join(map(str,r.tolist()))+' |\n'
    md+=f'\n**Macro average across {len(df)} held-out corpora**: MSA {mean_msa*100:.2f}%, Macro-F1 {mean_macro*100:.2f}%, UAR {mean_uar*100:.2f}%.\n'
    md+='\nTest uses 100% of target-corpus usable four-class recordings, no target adaptation. '
    md+='Source train/val are speaker-disjoint within each source corpus. '
    md+='PME4 may have excluded missing/duplicate audio. The six folds have separate model initialization.\n'
    (dest/'loco_results_table.md').write_text(md)
    # Publication-compatible LaTeX table (no results fabricated)
    tex=['\\begin{tabular}{lrrrrr}',r'\hline',
         r'Corpus & $N$ & MSA (\%) & Macro-F1 (\%) & UAR (\%) & Acc. (\%) \\',r'\hline']
    for r in df.itertuples():
        tex.append(f'{r.target_corpus} & {r.target_test_recordings} & {r.test_mean_subject_accuracy*100:.2f} & {r.test_macro_f1*100:.2f} & {r.test_uar*100:.2f} & {r.test_pooled_accuracy*100:.2f} \\\\')
    tex += [r'\hline',f'Macro mean & -- & {mean_msa*100:.2f} & {mean_macro*100:.2f} & {mean_uar*100:.2f} & -- \\\\',r'\hline',r'\end{tabular}']
    (dest/'loco_results_table.tex').write_text('\n'.join(tex))
    x=np.arange(len(df));w=.25
    fig,ax=plt.subplots(figsize=(11,5.5))
    for j,(key,label) in enumerate([('test_mean_subject_accuracy','Mean subject accuracy'),
                                    ('test_macro_f1','Macro-F1'),('test_uar','UAR')]):
        ax.bar(x+(j-1)*w,df[key]*100,w,label=label)
    ax.set_xticks(x,df.target_corpus,rotation=20)
    ax.set(ylim=(0,100),ylabel='Score (%)',title='WavLM-Large baseline: leave-one-corpus-out external results')
    ax.legend();fig.tight_layout();fig.savefig(dest/'loco_six_corpus_comparison.png',dpi=200);plt.close(fig)
    matrix=[]
    for corpus in df.target_corpus:
        classes=pd.read_csv(project/'folds'/FOLD_NAMES[corpus]/'external_test'/'per_class_metrics.csv')
        matrix.append([float(classes.loc[classes.emotion==k,'recall'].iloc[0]) for k in CLASS_NAMES])
    matrix=np.array(matrix)
    fig,ax=plt.subplots(figsize=(8,5.5));im=ax.imshow(matrix,cmap='Blues',vmin=0,vmax=1,aspect='auto')
    ax.set_xticks(range(4),CLASS_NAMES);ax.set_yticks(range(len(df)),df.target_corpus)
    for i in range(len(df)):
        for j in range(4):ax.text(j,i,f'{matrix[i,j]*100:.1f}%',ha='center',va='center',
                                      color='white' if matrix[i,j]>.6 else 'black')
    ax.set_title('External corpus: per-emotion recall (%)');fig.colorbar(im,ax=ax,label='Recall')
    fig.tight_layout();fig.savefig(dest/'loco_emotion_recall_heatmap.png',dpi=200);plt.close(fig)
    log('[PAPER]',len(df),'folds complete; mean external macro-F1',f'{mean_macro:.4%}',
        'MSA',f'{mean_msa:.4%}','saved',dest)


def smoke(args,project):
    # Does not download pretrained weights, trains no actual model, and makes
    # no synthetic "results" resembling external model performance.
    import wave
    from tempfile import TemporaryDirectory
    with TemporaryDirectory(prefix='loco_baseline_test_') as td:
        td=Path(td);data=td/'data';data.mkdir()
        sr=16000
        def write(p,seed):
            p.parent.mkdir(parents=True,exist_ok=True)
            t=np.arange(8000)/sr
            sig=np.sin((seed+100)*2*np.pi*t).astype(np.float32)*.15
            sf.write(p,sig,sr,subtype='PCM_16')
        # Provide each corpus with 8 speakers, four emotions; test the filename
        # decoding, RAVDESS nested duplicates, and PME4 CSV label assignment.
        raw=[]
        emotions=['NEU','SAD','ANG','HAP']
        for i in range(8):
            for j,emo in enumerate(emotions):
                write(data/'EAV'/f'subject{i+1}'/'Audio'/f'{2*j+2:03d}_Trial_02_Speaking_{CLASS_NAMES[j]}_aud.wav',i*40+j)
                write(data/'crema-d'/'AudioWAV'/f'{1001+i}_ABC_{emo}_XX.wav',1000+i*40+j)
                emo_code=['N','T','W','F'][j]
                # German IDs 01..08; sentence code 'a01'.
                write(data/'emodb'/'wav'/f'{i+1:02d}a01{emo_code}a.wav',2000+i*40+j)
                emo_code=['neu','tri','rab','gio'][j]
                write(data/'emovo'/'EMOVO'/f'{emo_code}-{("f" if i%2==0 else "m")}{i+1}-b1.wav',3000+i*40+j)
                src=data/'ravdess'/f'Actor_{i+1:02d}'/f'03-01-{["01","04","05","03"][j]}-01-01-01-{i+1:02d}.wav'
                write(src,4000+i*40+j)
                dst=data/'ravdess'/'audio_speech_actors_01-24'/f'Actor_{i+1:02d}'/src.name
                dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst)
                sp=data/'pme4'/'pme4-dataset'
                pp=sp/f's{i+1:02d}'/f's{i+1:02d}'/f't{j+1:03d}'/f's{i+1:02d}_t{j+1:03d}.wav'
                write(pp,5000+i*40+j)
                raw.append(dict(subject=i+1,trial=j+1,emotion=CLASS_NAMES[j],speech_start_time=.05,speech_stop_time=.3))
        import pandas as pd
        pd.DataFrame(raw).to_csv(data/'pme4'/'pme4-dataset'/'PME4_dataset_configs.csv',index=False)
        aa=argparse.Namespace(**vars(args));aa.data_root=str(data);aa.skip_sha=False
        df=discover(aa)
        assert len(df[df.status=='usable'])==6*8*4,len(df)
        # Reproduce actual CREMA-D fault: a WAV can have a valid header but
        # contain all-zero audio, which must be quarantined during AUDIT.
        bad_file=data/'crema-d'/'AudioWAV'/'1001_ABC_SAD_XX.wav'
        sf.write(bad_file,np.zeros(8000,dtype=np.float32),sr,subtype='PCM_16')
        df_bad=discover(aa)
        bad_rec=df_bad[df_bad.recording_id=='CREMA-D/1001_ABC_SAD_XX']
        assert len(bad_rec)==1 and bad_rec.status.iloc[0]=='invalid_audio_signal',bad_rec.to_dict('records')
        assert len(df_bad[df_bad.status=='usable'])==6*8*4-1
        log('[SMOKE OK] header-valid but silent CREMA-D audio quarantined before split')
        ss=choose_source_val_speakers(df,seed=2026)
        for target in CORPORA:
            fm=make_fold_manifest(df,ss,target)
            assert len(fm[fm.split=='test'])==32
            assert fm[fm.split=='train'].dataset.nunique()==5
            assert fm[fm.split=='val'].dataset.nunique()==5
        rec=df[df.status=='usable'].head(9).reset_index(drop=True)
        rec['cache_path']=[str(p) for p in rec.file_path]
        rec['processed_samples']=8000
        # Basic tensor contracts using synthetic in-memory array, without
        # interpreting any training metric as real research result.
        for size in (8000,64000,101337):
            assert eval_positions(size,64000,5)
        probs=np.eye(4,dtype='float32')[rec.class_id.to_numpy()]
        m,subs,cl,sc,d,yp=metric_summary(rec,probs)
        assert m['mean_subject_accuracy']==1.0
        assert m['n_subjects']==len(subs)
        sampler_df=pd.DataFrame([dict(dataset=c,class_id=i) for c in CORPORA if c!='EAV' for i in range(4)])
        sample=CorpusEmotionSampler(sampler_df,160)
        assert len(list(iter(sample)))==160
        log('[SMOKE OK] six corpus parsers, duplicated RAVDESS suppression, PME4 metadata,')
        log('[SMOKE OK] six LOCO folds, source speaker disjointness, subject-level averaging, balanced sampler')
        log('[SMOKE NOTE] No real WavLM weights were loaded, and no empirical accuracy was measured.')


def preflight(args,project):
    set_seed(args.seed)
    device=torch.device('cuda' if torch.cuda.is_available() and args.device!='cpu' else 'cpu')
    model=get_pretrained_model(args).to(device)
    model.set_stage('finetune');model.train()
    ns=int(args.crop_seconds*16000)
    x=torch.randn(2,ns,device=device)
    mask=torch.ones(2,ns,dtype=torch.long,device=device)
    mask[1,ns//2:]=0;x[1,ns//2:]=0
    with amp_ctx(device):
        z,emb=model(x,mask)
        loss=nn.functional.cross_entropy(z,torch.tensor([0,1],device=device))
    loss.backward()
    assert z.shape==(2,4) and emb.shape==(2,1024)
    assert torch.isfinite(loss).item()
    assert any(p.grad is not None for p in model.wavlm.encoder.layers[-1].parameters())
    log('[PREFLIGHT OK]',device,'logits',list(z.shape),'pooled',list(emb.shape),
        'loss',float(loss.item()),'trainable',sum(p.numel() for p in model.parameters() if p.requires_grad))
    del model;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()


def main():
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--mode',choices=['smoke','audit','prepare','preflight','train','test','aggregate','full'],default='audit')
    parser.add_argument('--data-root',default=str(DATA_DEFAULT))
    parser.add_argument('--project-root',default=str(PROJECT_DEFAULT))
    parser.add_argument('--target',default='all',choices=['all',*CORPORA],help='Target corpus to hold out; "all" means six folds')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--skip-sha',action='store_true',help='Skip SHA256 checks (not recommended for research)')
    parser.add_argument('--rebuild-audit',action='store_true',help='Rebuild audit ONLY before training started')
    parser.add_argument('--min-audio-seconds',type=float,default=.15)
    parser.add_argument('--pme4-speech-trim',dest='pme4_speech_trim',action='store_true',default=True)
    parser.add_argument('--no-pme4-speech-trim',dest='pme4_speech_trim',action='store_false')
    parser.add_argument('--trim-context',type=float,default=.15,help='Context padding around PME4 speech annotations (seconds)')
    parser.add_argument('--crop-seconds',type=float,default=4.0)
    parser.add_argument('--eval-windows',type=int,default=6)
    parser.add_argument('--epochs',type=int,default=28)
    parser.add_argument('--freeze-epochs',type=int,default=4)
    parser.add_argument('--finetune-layers',type=int,default=6)
    parser.add_argument('--patience',type=int,default=7)
    parser.add_argument('--min-delta',type=float,default=.002)
    parser.add_argument('--microbatch',type=int,default=4)
    parser.add_argument('--grad-accum',type=int,default=4)
    parser.add_argument('--eval-batch',type=int,default=6)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--head-lr-frozen',type=float,default=3e-4)
    parser.add_argument('--head-lr',type=float,default=1e-4)
    parser.add_argument('--backbone-lr',type=float,default=5e-6)
    parser.add_argument('--layer-decay',type=float,default=.85)
    parser.add_argument('--weight-decay',type=float,default=.01)
    parser.add_argument('--warmup-frac',type=float,default=.05)
    parser.add_argument('--dropout',type=float,default=.2)
    parser.add_argument('--progress-batches',type=int,default=100)
    parser.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--unlock-test',action='store_true',help='Required for any held-out target corpus prediction')
    parser.add_argument('--force-retest',action='store_true',help='Not recommended; will recompute previously sealed test scores')
    args=parser.parse_args()
    if args.seed<0 or args.workers<0 or args.epochs<=args.freeze_epochs or args.freeze_epochs<1:
        parser.error('Invalid seed/workers/epochs/freeze-epochs settings')
    if args.microbatch<1 or args.grad_accum<1 or args.eval_batch<1 or args.eval_windows<1 or not 0<args.crop_seconds<=16:
        parser.error('Invalid batch/window settings')
    if args.finetune_layers<1 or args.finetune_layers>24:
        parser.error('Finetune top layers must be 1..24')
    project=safe_setup(args)
    if args.mode=='smoke':smoke(args,project);return
    if args.mode=='preflight':preflight(args,project);return
    if args.mode=='aggregate':aggregate(project);return
    audit,speakers,sha=get_audit(args,project,force=args.rebuild_audit)
    if args.mode=='audit':
        log('Audit complete. Next run --mode prepare.');return
    if args.mode=='prepare':
        prepare_audio(args,project,audit);return
    mapping=cache_manifest(args,project,audit)
    targets=list(CORPORA) if args.target=='all' else [args.target]
    if args.mode=='full' and not args.unlock_test:
        raise RuntimeError('--mode full includes external test; explicitly supply --unlock-test, or use train/test separately.')
    if args.mode in ('train','full') and args.device!='cpu' and not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU unavailable. Verify the A100 with nvidia-smi, or explicitly use --device cpu for debug.')
    if args.mode in ('train','full'):
        for target in targets:train_fold(args,project,audit,speakers,mapping,sha,target)
    if args.mode in ('test','full'):
        if not args.unlock_test:raise RuntimeError('TEST EMBARGOED: supply --unlock-test')
        if args.target=='all':
            missing=[c for c in CORPORA if not (project/'folds'/FOLD_NAMES[c]/'training_complete.json').is_file()]
            if missing:raise RuntimeError(f'Cannot evaluate all test corpora until all six trainings finished: {missing}')
        for target in targets:test_fold(args,project,audit,speakers,mapping,sha,target)
        aggregate(project)
    log('[COMPLETE]',args.mode,'outputs',project)


if __name__=='__main__':
    main()
