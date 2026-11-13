#!/usr/bin/env python3
# SAFE EVALUATION PATCH (2026-11): no change to architecture, model parameters,
# trained checkpoints, input crops, dataset manifests or checkpoint-selection logic.
# Default inference DataLoader uses the main process; CPU workers are still used for training.
# --recover-crashed-test archives lock ONLY when no saved test outputs exist.
"""Architecture A: WavLM-Large + GHF + DPTP + supervised contrastive learning.

FOUR STRICT LEAVE-ONE-CORPUS-OUT (LOCO) SPEECH EMOTION EXPERIMENTS:
  TRAIN: three complete source corpora, apart from their speaker-disjoint validation speakers.
  VALIDATE: held-out speakers from EACH of those three source corpora.
  TEST: the entire fourth corpus, never used for model selection.

Shared six-class ontology: Neutral, Sad, Angry, Happy, Fear, Disgust.
- RAVDESS calm and surprise excluded; speech-only; avoid double-importing nested copies.
A FULL-SCAN DATA INTEGRITY AUDIT runs over every eligible audio file in all four
corpora. Benign exact repeats are deduplicated; contradictory label, cross-speaker,
cross-corpus, and conflicting-identity groups are quarantined in their ENTIRETY.
Every exclusion is reported. The audit never stops at the first bad WAV or group.
No quarantined audio can enter training, source validation, or unseen-corpus tests.
- EmoDB boredom excluded; EMOVO surprise excluded.
- CREMA-D uses all six emotion labels.

Reference architecture: uploaded PME4 Architecture A (24-layer WavLM-Large, 4 x 6-layer
Grouped Hierarchical Fusion, 8-head Dual-Path Temporal Pooling, 512-d embedding,
128-d SupCon projection, freeze then fine-tune WavLM's top six transformer layers).

HIGH-THROUGHPUT A100 PRESET: 13.0 GiB process target (11.0 GiB PyTorch allocator
cap + 2.0 GiB CUDA headroom). Frozen-stage microbatch 12 x accumulation 2;
finetuning microbatch 6 x accumulation 4. Evaluation microbatch 12. Effective
batch=24 and 150 optimizer steps per epoch in BOTH stages. Four preprocessing
workers and pinned-memory transfers. This is an engineering target, NOT a
hardware VRAM quota; use MIG for strict isolation. The larger finetuning batch
uses within-microbatch SupCon instead of the two-example CPU queue, so the
numerical training objective is NOT identical to the 2-example run. The same
WavLM-Large, GHF, DPTP, CE+SupCon principle and LOCO research protocol remain.

For safe resume of an existing 2x12 run at the NEXT completed epoch boundary,
pass --resume --allow-throughput-resume --output-dir <original_output_dir>.
This explicitly records changed batch and VRAM configuration; it does not
recover partial epochs or claim bitwise identical training.

Requires Python 3.10+, NVIDIA BF16 GPU for train/test, torch, transformers==4.46.3,
numpy, pandas, scipy, soundfile, scikit-learn, matplotlib, tqdm, safetensors.
Install examples:
  pip install 'transformers==4.46.3' 'huggingface_hub>=0.26,<1' \
     'soundfile>=0.12' 'scipy>=1.11' 'pandas>=2' 'scikit-learn>=1.3' \
     'matplotlib>=3.7' 'tqdm>=4.65' safetensors
  # Install the correct NVIDIA CUDA PyTorch separately, if needed.

ONE-COMMAND FOUR-EXPERIMENT RUN (does not bypass one-time test guard):
  python cross_corpus_architecture_a_fast_15gb.py --mode all --unlock-test

Separate training + testing:
  python cross_corpus_architecture_a_fast_15gb.py --mode audit
  python cross_corpus_architecture_a_fast_15gb.py --mode train
  python cross_corpus_architecture_a_fast_15gb.py --mode test --unlock-test
  python cross_corpus_architecture_a_fast_15gb.py --mode report

Resume interrupted training:
  python cross_corpus_architecture_a_fast_15gb.py --mode all --unlock-test --resume

All command examples assume datasets in /workspace/data and the script is located in
/workspace/avyakt-sc-final/sc-final-speech-model/.

No access to the remote A100/data is implied: results only exist after you run it.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import softmax
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             log_loss, precision_recall_curve, precision_recall_fscore_support,
                             roc_auc_score, roc_curve)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

CORPORA = ('ravdess', 'crema-d', 'emodb', 'emovo')
LABELS = ('Neutral', 'Sad', 'Angry', 'Happy', 'Fear', 'Disgust')
SLUGS = tuple(s.lower() for s in LABELS)
CLASS_TO_ID = {s: i for i, s in enumerate(SLUGS)}
NC = len(LABELS)
SAMPLE_RATE = 16000
PRETRAINED_MODEL = 'microsoft/wavlm-large'
# The exact revision pinned by the supplied PME4 reference script.
DEFAULT_REVISION = 'c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
PROTOCOL = 'ARCH_A_LOCO_SIX_CLASS_SPEAKER_DISJOINT_SOURCE_VAL_V1'
AUDIT_POLICY_VERSION = 'FULL_SCAN_QUARANTINE_AMBIGUOUS_GROUPS_V2'
FORMAT = 'ARCH_A_LOCO_6CLASS_CHECKPOINT_V1'

RAVDESS_CODE = {'01': 'neutral', '03': 'happy', '04': 'sad', '05': 'angry',
                '06': 'fear', '07': 'disgust'}
CREMA_CODE = {'NEU': 'neutral', 'SAD': 'sad', 'ANG': 'angry',
              'HAP': 'happy', 'FEA': 'fear', 'DIS': 'disgust'}
EMODB_CODE = {'N': 'neutral', 'T': 'sad', 'W': 'angry',
              'F': 'happy', 'A': 'fear', 'E': 'disgust'}
EMOVO_CODE = {'neu': 'neutral', 'tri': 'sad', 'rab': 'angry',
              'gio': 'happy', 'pau': 'fear', 'dis': 'disgust'}


def args_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--mode', choices=['audit', 'train', 'test', 'all', 'report', 'self-test'], default='all')
    p.add_argument('--data-root', type=Path, default=Path('/workspace/data'))
    p.add_argument('--output-dir', type=Path,
                   default=Path('/workspace/avyakt-sc-final/sc-final-speech-model/outputs/cross_corpus_architecture_a_fast_15gb'))
    p.add_argument('--experiment', choices=['all', *CORPORA], default='all',
                   help='Select test corpus or run all four experiments')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--val-speaker-fraction', type=float, default=0.20)
    p.add_argument('--window-seconds', type=float, default=3.0)
    p.add_argument('--eval-hop-fraction', type=float, default=0.5)
    p.add_argument('--max-eval-windows', type=int, default=5)
    p.add_argument('--batch-size', type=int, choices=[2, 6, 12], default=6,
                   help='Fine-tune microbatch (default 6). Effective batch stays 24.')
    p.add_argument('--frozen-batch-size', type=int, choices=[2, 6, 12], default=12,
                   help='Frozen-stage microbatch (default 12). Effective batch stays 24.')
    p.add_argument('--grad-accum', type=int, default=None)
    p.add_argument('--eval-batch-size', type=int, default=12)
    p.add_argument('--max-gpu-gib', type=float, default=13.0,
                   help='Target <=13 GiB process memory; verify against your actual free VRAM.')
    p.add_argument('--cuda-overhead-gib', type=float, default=2.0,
                   help='Reserved headroom for CUDA context/non-PyTorch allocations (default 2 GiB).')
    p.add_argument('--contrastive-queue-size', type=int, default=144,
                   help='Detached projection keys (stored on CPU), to provide negatives with batch size 2.')
    p.add_argument('--updates-per-epoch', type=int, default=150,
                   help='Fixed balanced optimizer updates per virtual epoch.')
    p.add_argument('--max-epochs', type=int, default=28)
    p.add_argument('--frozen-epochs', type=int, default=4)
    p.add_argument('--gradient-checkpointing', action=argparse.BooleanOptionalAction, default=False,
                   help='Off for speed: top 6 trainable layers generally fit within 11 GiB allocator cap. Enable after OOM.')
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--min-delta', type=float, default=0.002)
    p.add_argument('--num-workers', type=int, default=4,
                   help='Data-loading CPU workers (default 4; use 0 if Docker shared memory is limited).')
    p.add_argument('--inference-batch-size', type=int, default=None,
                   help='Evaluation-only microbatch override, for safe fallback without changing trained config.')
    p.add_argument('--eval-workers', type=int, default=0,
                   help='Evaluation-only DataLoader workers. Default 0 avoids native fork/UCX crashes; training worker count unchanged.')
    p.add_argument('--recover-crashed-test', action='store_true',
                   help='Explicitly retry a test crashed before ANY predictions/results were saved; archives the stale lock.')
    p.add_argument('--cache-audio', action=argparse.BooleanOptionalAction, default=True)
    p.add_argument('--skip-wav-hash', action='store_true',
                   help='Audit-only unsafe option: disables content hashes; training/test refuses it.')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--allow-throughput-resume', action='store_true',
                   help='Explicitly permit changing ONLY batch/memory settings when resuming an old run; logged for reproducibility.')
    p.add_argument('--unlock-test', action='store_true',
                   help='Required for --mode test/all; one-time test per held-out corpus.')
    p.add_argument('--checkpoint-revision', default=DEFAULT_REVISION)
    p.add_argument('--max-file-seconds', type=float, default=30.0)
    p.add_argument('--min-file-seconds', type=float, default=0.3)
    p.add_argument('--bootstrap-repeats', type=int, default=5000)
    a = p.parse_args()
    if a.grad_accum is None:
        a.grad_accum = {2: 12, 6: 4, 12: 2}[a.batch_size]
    if a.batch_size * a.grad_accum != 24:
        p.error('Effective fine-tuning batch must be 24: 2x12, 6x4, or 12x2')
    a.frozen_grad_accum = 24 // a.frozen_batch_size
    if a.allow_throughput_resume and not a.resume:
        p.error('--allow-throughput-resume requires --resume')
    if not (3.0 <= a.max_gpu_gib <= 80.0) or not (0.5 <= a.cuda_overhead_gib < a.max_gpu_gib-0.5):
        p.error('Invalid GPU budget; require 3 <= max-gpu-gib <= 80 and 0.5 <= overhead < budget-0.5')
    if a.contrastive_queue_size < 6:
        p.error('--contrastive-queue-size must be at least 6')
    if any((a.updates_per_epoch*accum) % (12//batch) for batch, accum in
           ((a.batch_size, a.grad_accum), (a.frozen_batch_size, a.frozen_grad_accum))):
        p.error('Updates/accumulation must cover whole six-emotion sampling cycles in both stages')
    if not (0 < a.val_speaker_fraction <= 0.40):
        p.error('--val-speaker-fraction must be in (0, 0.40]')
    if a.window_seconds < 1.0 or a.window_seconds > 8.0:
        p.error('Window seconds must be in [1, 8]')
    if not (0 < a.eval_hop_fraction <= 1):
        p.error('Hop fraction must be in (0, 1]')
    if a.max_eval_windows < 1 or a.eval_batch_size < 1 or (a.inference_batch_size is not None and a.inference_batch_size < 1) or a.num_workers < 0 or a.eval_workers < 0:
        p.error('Invalid evaluation/worker configuration')
    if a.max_epochs <= a.frozen_epochs or a.frozen_epochs < 1 or a.patience < 1:
        p.error('Must have 1 <= frozen-epochs < max-epochs and patience >= 1')
    if a.updates_per_epoch < 1 or a.bootstrap_repeats < 100:
        p.error('Need >=1 optimizer update and >=100 bootstrap replicates')
    if a.mode in ('all', 'test') and not a.unlock_test:
        p.error('Held-out corpus testing is locked. Supply --unlock-test explicitly.')
    if not re.fullmatch('[a-f0-9]{40}', a.checkpoint_revision):
        p.error('--checkpoint-revision must be a pinned 40-hex commit SHA')
    return a


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def classify_duplicate_component(items):
    """Classify a connected duplicate group WITHOUT depending on encounter order.

    Union edges are generated by identical recording identity, full-WAV SHA-256,
    or SHA-256 of the *decoded mono float32* signal at its original sample rate.
    A mixed-label/speaker/corpus group is removed entirely; retaining the first
    item would allow arbitrary conflicting labels or test leakage.
    """
    reasons = []
    if len({r['corpus'] for r in items}) > 1:
        reasons.append('cross_corpus_audio_overlap')
    if len({r['speaker_id'] for r in items}) > 1:
        reasons.append('cross_speaker_audio_overlap')
    if len({r['emotion'] for r in items}) > 1:
        reasons.append('conflicting_emotion_labels')
    by_identity = defaultdict(list)
    for r in items:
        by_identity[r['recording_id']].append(r)
    if any(len({(v['sha256'], v['pcm_sha256']) for v in group}) > 1
           for group in by_identity.values()):
        reasons.append('same_recording_id_different_audio')
    if reasons:
        return 'quarantine_all', '+'.join(reasons)
    return 'retain_one', 'redundant_same_corpus_speaker_emotion'


def resolve_duplicate_components(all_records, hashing_enabled=True):
    """Return kept, excluded, and group reports after scanning ALL four datasets.

    The union-find ensures that a later third file can invalidate a formerly safe
    two-file group. This prevents the subtle bug of keeping a conflicting first
    copy when the conflict is found only after it was inserted into the manifest.
    """
    records = sorted(all_records, key=lambda r: (r['corpus'], r['recording_id'], r['abs_path']))
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        x, y = find(a), find(b)
        if x != y:
            parent[max(x, y)] = min(x, y)

    lookup = {}
    for i, r in enumerate(records):
        keys = [('id', r['recording_id'])]
        if hashing_enabled:
            keys.extend((('wav_sha256', r['sha256']), ('pcm_sha256', r['pcm_sha256'])))
        for key in keys:
            if key in lookup:
                union(i, lookup[key])
            else:
                lookup[key] = i

    groups = defaultdict(list)
    for i in range(len(records)):
        groups[find(i)].append(records[i])
    keep, dropped, group_rows, group_details = [], [], [], []
    actions = {}
    for group in sorted(groups.values(), key=lambda g: min(r['abs_path'] for r in g)):
        group.sort(key=lambda r: (r['corpus'], r['recording_id'], r['abs_path']))
        if len(group) == 1:
            r = group[0]
            keep.append(r)
            actions[r['abs_path']] = ('retained', 'unique_verified_audio')
            continue
        action, reason = classify_duplicate_component(group)
        group_id = hashlib.sha256(json.dumps(
            [(r['recording_id'], r['abs_path'], r['sha256'], r['pcm_sha256']) for r in group],
            separators=(',', ':'), sort_keys=True).encode()).hexdigest()[:20]
        representative = group[0] if action == 'retain_one' else None
        if representative is not None:
            keep.append(representative)
            actions[representative['abs_path']] = ('retained', 'canonical_copy')
        for r in (group[1:] if representative is not None else group):
            status = 'redundant_duplicate' if representative is not None else 'quarantined_conflict'
            actions[r['abs_path']] = (status, reason)
            dropped.append({
                'group_id': group_id, 'action': status, 'reason': reason,
                'corpus': r['corpus'], 'recording_id': r['recording_id'],
                'speaker_id': r['speaker_id'], 'emotion': r['emotion'],
                'sha256': r['sha256'], 'pcm_sha256': r['pcm_sha256'],
                'path': r['abs_path'], 'kept_recording_id':
                    representative['recording_id'] if representative else '',
                'kept_path': representative['abs_path'] if representative else ''})
        group_rows.append({
            'group_id': group_id, 'action': action, 'reason': reason,
            'total_files': len(group), 'files_removed': len(group)-(1 if representative else 0),
            'corpora': ';'.join(sorted({r['corpus'] for r in group})),
            'speakers': ';'.join(sorted({r['speaker_id'] for r in group})),
            'emotions': ';'.join(sorted({r['emotion'] for r in group})),
            'recording_ids': ';'.join(sorted({r['recording_id'] for r in group}))})
        group_details.append({'group_id': group_id, 'action': action, 'reason': reason,
                              'files': [{k: r[k] for k in ('corpus', 'recording_id', 'speaker_id',
                                     'emotion', 'abs_path', 'sha256', 'pcm_sha256')} for r in group]})
    return keep, dropped, group_rows, group_details, actions


def stable_seed(*parts):
    h = hashlib.sha256('|'.join(map(str, parts)).encode()).digest()
    return int.from_bytes(h[:8], byteorder='little', signed=False)


def make_logger(output):
    output.mkdir(parents=True, exist_ok=True)
    log = logging.getLogger('ARCH_A_CROSS_CORPUS')
    log.handlers.clear()
    log.setLevel(logging.INFO)
    log.propagate = False
    fmt = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', '%Y-%m-%d %H:%M:%S')
    for handler in (logging.StreamHandler(sys.stderr), logging.FileHandler(output/'experiment.log')):
        handler.setFormatter(fmt)
        log.addHandler(handler)
    return log


def parse_record(corpus: str, path: Path):
    """Return (speaker_id, canonical emotion) or None for explicitly excluded labels."""
    stem = path.stem
    if corpus == 'ravdess':
        m = re.fullmatch(r'(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)-(\d\d)', stem)
        if m is None:
            raise ValueError(f'Unrecognized RAVDESS filename: {path.name}')
        modality, channel, emo, intensity, statement, repetition, actor = m.groups()
        if modality != '03' or channel != '01':
            return None  # audio-only speech, not song
        if emo in ('02', '08'):
            return None  # calm and surprise are not shared across all four corpora
        if emo not in RAVDESS_CODE:
            raise ValueError(f'Unknown RAVDESS code: {path.name}')
        if actor not in {f'{n:02d}' for n in range(1, 25)}:
            raise ValueError(f'Unknown actor: {path.name}')
        if intensity not in ('01', '02') or statement not in ('01', '02') or repetition not in ('01', '02'):
            raise ValueError(f'Invalid filename codes: {path.name}')
        return actor, RAVDESS_CODE[emo]
    if corpus == 'crema-d':
        m = re.fullmatch(r'(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})', stem)
        if not m:
            raise ValueError(f'Unrecognized CREMA-D filename: {path.name}')
        actor, sentence, emo, level = m.groups()
        if emo not in CREMA_CODE or level not in ('XX', 'LO', 'MD', 'HI'):
            raise ValueError(f'Invalid CREMA-D emotion/intensity: {path.name}')
        return actor, CREMA_CODE[emo]
    if corpus == 'emodb':
        m = re.fullmatch(r'(\d\d)([ab]\d\d)([A-Z])([a-z])', stem)
        if not m:
            raise ValueError(f'Unrecognized EmoDB filename: {path.name}')
        actor, statement, emo, version = m.groups()
        if emo == 'L':
            return None  # boredom excluded
        if emo not in EMODB_CODE:
            raise ValueError(f'Unknown EmoDB emotion: {path.name}')
        return actor, EMODB_CODE[emo]
    if corpus == 'emovo':
        m = re.fullmatch(r'([a-z]{3})-([fm][123])-([bldn]\d+)', stem.lower())
        if not m:
            raise ValueError(f'Unrecognized EMOVO filename: {path.name}')
        emo, actor, statement = m.groups()
        if emo == 'sor':
            return None  # surprise excluded
        if emo not in EMOVO_CODE:
            raise ValueError(f'Unknown EMOVO code: {path.name}')
        if path.parent.name.lower() != actor:
            raise ValueError(f'EMOVO speaker folder/filename disagreement: {path}')
        return actor, EMOVO_CODE[emo]
    raise ValueError(f'Unknown corpus: {corpus}')


def paths_for(corpus, root):
    if corpus == 'crema-d':
        folder = root/corpus/'AudioWAV'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing CREMA-D AudioWAV folder: {folder}')
        return sorted(folder.glob('*.wav'))
    if corpus == 'emodb':
        folder = root/corpus/'wav'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing EmoDB wav folder: {folder}')
        return sorted(folder.glob('*.wav'))
    if corpus == 'emovo':
        folder = root/corpus/'EMOVO'
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing EMOVO folder: {folder}')
        return sorted(p for p in folder.glob('*/*.wav') if p.parent.name.lower() in
                      ('f1', 'f2', 'f3', 'm1', 'm2', 'm3'))
    if corpus == 'ravdess':
        folder = root/corpus
        if not folder.is_dir():
            raise FileNotFoundError(f'Missing RAVDESS folder: {folder}')
        # Scan BOTH the canonical Actor_XX/ directories and extra nested copies.
        # Duplicate IDs with different contents are quarantined rather than
        # raising an exception halfway through an inventory scan.
        return sorted(folder.rglob('*.wav'))
    raise ValueError(corpus)


def waveform_integrity_check(path):
    """Validate audio and hash the decoded mono audio used for model input.

    Comparing both original WAV bytes and decoded audio catches files that have
    different headers/encodings but exactly the same sample content. The hash
    includes original sampling rate so unlike-rate waveforms are not falsely
    identified as sample-exact duplicates; it does not detect near duplicates.
    """
    energy, count, peak = 0., 0, 0.
    digest = hashlib.sha256()
    with sf.SoundFile(path, mode='r') as infile:
        digest.update(f'{infile.samplerate}:mono-float32:'.encode())
        while True:
            block = infile.read(65536, dtype='float32', always_2d=True)
            if block.size == 0:
                break
            if not np.isfinite(block).all():
                raise ValueError('Audio contains NaN/Inf')
            mono = block.mean(axis=1, dtype=np.float32)
            peak = max(peak, float(np.max(np.abs(mono))))
            energy += float(np.dot(mono.astype(np.float64), mono.astype(np.float64)))
            count += len(mono)
            digest.update(mono.astype('<f4', copy=False).tobytes())
    if count == 0 or peak < 1e-7 or math.sqrt(energy/count) < 1e-7:
        raise ValueError('Empty or near-silent audio')
    return digest.hexdigest()


def audit_corpora(a, log):
    """Exhaustively audit every eligible WAV before choosing any train/val split.

    Invalid files and ambiguous duplicate groups are excluded with detailed
    provenance; the full corpus is scanned before reporting viability failures.
    The raw datasets are never modified, and target/test labels are not used for
    optimization or checkpoint selection.
    """
    root = a.data_root.resolve()
    output = a.output_dir.resolve()
    audit_dir = output/'audit'
    audit_dir.mkdir(parents=True, exist_ok=True)
    candidates, rejects, excluded, file_inventory = [], [], Counter(), []
    scan_errors, scanned_files = [], Counter()
    for corpus in CORPORA:
        try:
            paths = paths_for(corpus, root)
        except (FileNotFoundError, OSError, ValueError) as exc:
            scan_errors.append(f'{corpus}: {exc}')
            log.error('[AUDIT] %s: %s; continuing with remaining corpora', corpus, exc)
            continue
        if not paths:
            scan_errors.append(f'{corpus}: no WAVs discovered')
            log.error('[AUDIT] %s: no WAV files; continuing with remaining corpora', corpus)
            continue
        log.info('[AUDIT] %s: %d WAV paths (including any re-extracted copies)', corpus, len(paths))
        for p in tqdm(paths, desc=f'Audit {corpus}', mininterval=1.5,
                      disable=not sys.stderr.isatty()):
            scanned_files[corpus] += 1
            inv = {'corpus': corpus, 'path': str(p.resolve()), 'recording_id': f'{corpus}:{p.stem}',
                   'speaker_id': '', 'emotion': '', 'status': '', 'reason': ''}
            try:
                parsed = parse_record(corpus, p)
                if parsed is None:
                    excluded[corpus] += 1
                    inv.update(status='excluded_label_or_non_speech',
                               reason='outside_shared_six_class_protocol')
                    file_inventory.append(inv)
                    continue
                sid, emotion = parsed
                speaker = f'{corpus}:{sid}'
                inv.update(speaker_id=speaker, emotion=emotion)
                if corpus == 'ravdess' and p.parent.name.startswith('Actor_'):
                    if p.parent.name != f'Actor_{sid}':
                        raise ValueError('RAVDESS speaker directory contradicts filename')
                header = sf.info(p)
                if header.channels < 1 or header.samplerate < 4000 or header.frames < 1:
                    raise ValueError('Invalid WAV audio header')
                duration = header.frames/header.samplerate
                if not a.min_file_seconds <= duration <= a.max_file_seconds:
                    raise ValueError(f'duration={duration:.3f}s outside configured range')
                pcm_hash = waveform_integrity_check(p)
                sha = digest_file(p) if not a.skip_wav_hash else 'HASH_DISABLED'
                candidates.append(dict(corpus=corpus, recording_id=inv['recording_id'],
                            speaker_id=speaker, emotion_id=CLASS_TO_ID[emotion], emotion=emotion,
                            rel_path=str(p.relative_to(root)), abs_path=str(p.resolve()),
                            orig_sr=int(header.samplerate), orig_frames=int(header.frames),
                            orig_channels=int(header.channels), duration_s=float(duration),
                            sha256=sha, pcm_sha256=pcm_hash if not a.skip_wav_hash else 'HASH_DISABLED'))
                inv.update(status='eligible')
            except (OSError, RuntimeError, ValueError, EOFError, TypeError) as exc:
                # Do not stop at the first unparsable, unreadable, truncated, or
                # contradictory WAV; inventory every problem across four corpora.
                detail = f'{type(exc).__name__}: {exc}'
                rejects.append({'corpus': corpus, 'path': str(p.resolve()),
                                'recording_id': inv['recording_id'], 'reason': detail})
                inv.update(status='rejected_invalid_file', reason=detail)
            file_inventory.append(inv)
        log.info('[AUDIT] %s scanned=%d eligible_before_dedup=%d rejected=%d skipped_labels=%d',
                 corpus, scanned_files[corpus], sum(r['corpus'] == corpus for r in candidates),
                 sum(r['corpus'] == corpus for r in rejects), excluded[corpus])

    kept, duplicates, groups, group_details, actions = resolve_duplicate_components(
        candidates, hashing_enabled=not a.skip_wav_hash)
    for inv in file_inventory:
        if inv['status'] == 'eligible':
            status, reason = actions[inv['path']]
            inv.update(status=status, reason=reason)

    # Write every diagnostic before any viability check, so users receive ONE
    # complete report rather than an exception for each individual bad WAV.
    pd.DataFrame(file_inventory, columns=['corpus', 'path', 'recording_id', 'speaker_id',
        'emotion', 'status', 'reason']).to_csv(audit_dir/'full_scan_inventory.csv', index=False)
    pd.DataFrame(rejects, columns=['corpus', 'path', 'recording_id', 'reason']).to_csv(
        audit_dir/'rejected_audio.csv', index=False)
    duplicate_cols = ['group_id', 'action', 'reason', 'corpus', 'recording_id',
                      'speaker_id', 'emotion', 'sha256', 'pcm_sha256', 'path',
                      'kept_recording_id', 'kept_path']
    pd.DataFrame(duplicates, columns=duplicate_cols).to_csv(
        audit_dir/'duplicate_audio.csv', index=False)
    pd.DataFrame(groups, columns=['group_id', 'action', 'reason', 'total_files',
                                  'files_removed', 'corpora', 'speakers', 'emotions',
                                  'recording_ids']).to_csv(audit_dir/'duplicate_groups.csv', index=False)
    pd.DataFrame([r for r in duplicates if r['action'] == 'quarantined_conflict'],
                 columns=duplicate_cols).to_csv(audit_dir/'quarantined_audio.csv', index=False)
    write_json(audit_dir/'duplicate_group_details.json', group_details)

    df = pd.DataFrame(kept, columns=['corpus', 'recording_id', 'speaker_id',
            'emotion_id', 'emotion', 'rel_path', 'abs_path', 'orig_sr',
            'orig_frames', 'orig_channels', 'duration_s', 'sha256', 'pcm_sha256'])
    if not df.empty:
        df = df.sort_values(['corpus', 'recording_id']).reset_index(drop=True)
        assert not df.recording_id.duplicated().any(), 'BUG: duplicate identity survived quarantine'
        if not a.skip_wav_hash:
            assert not df.sha256.duplicated().any(), 'BUG: exact WAV duplicate survived quarantine'
            assert not df.pcm_sha256.duplicated().any(), 'BUG: decoded PCM duplicate survived quarantine'
    df.to_csv(audit_dir/'all_corpora_manifest.csv', index=False)
    histogram = (df.groupby(['corpus', 'emotion'], observed=True).size().unstack(fill_value=0)
                 if len(df) else pd.DataFrame())
    histogram = histogram.reindex(index=list(CORPORA), columns=list(SLUGS), fill_value=0)
    histogram.to_csv(audit_dir/'class_counts_by_corpus.csv')
    pre = pd.DataFrame(candidates)
    counts_before = (pre.groupby(['corpus', 'emotion']).size().unstack(fill_value=0)
                     if len(pre) else pd.DataFrame())
    counts_before = counts_before.reindex(index=list(CORPORA), columns=list(SLUGS), fill_value=0)
    comparison = []
    for c in CORPORA:
        for emotion in SLUGS:
            initial, remaining = int(counts_before.loc[c, emotion]), int(histogram.loc[c, emotion])
            comparison.append({'corpus': c, 'emotion': emotion,
                               'eligible_before_dedup': initial,
                               'retained_after_dedup': remaining,
                               'removed': initial-remaining,
                               'removed_percent': 100*(initial-remaining)/max(1, initial)})
    pd.DataFrame(comparison).to_csv(audit_dir/'class_counts_before_after_dedup.csv', index=False)
    specs = (df.groupby('corpus').agg(n_audio=('recording_id', 'size'),
              speakers=('speaker_id', 'nunique'), duration_mean=('duration_s', 'mean'),
              duration_median=('duration_s', 'median'),
              duration_p10=('duration_s', lambda x: x.quantile(.1)),
              duration_p90=('duration_s', lambda x: x.quantile(.9))).reset_index()
              if len(df) else pd.DataFrame(columns=['corpus', 'n_audio', 'speakers',
                  'duration_mean', 'duration_median', 'duration_p10', 'duration_p90']))
    specs.to_csv(audit_dir/'corpus_audio_statistics.csv', index=False)

    quarantined = [r for r in duplicates if r['action'] == 'quarantined_conflict']
    redundant = [r for r in duplicates if r['action'] == 'redundant_duplicate']
    integrity_summary = {}
    warnings = []
    for c in CORPORA:
        eligible = sum(r['corpus'] == c for r in candidates)
        remain = int((df.corpus == c).sum())
        qc = sum(r['corpus'] == c for r in quarantined)
        rd = sum(r['corpus'] == c for r in redundant)
        integrity_summary[c] = dict(scanned_files=int(scanned_files[c]),
            excluded_unshared_labels=int(excluded[c]),
            rejected_audio=int(sum(r['corpus'] == c for r in rejects)),
            eligible_before_duplicates=int(eligible),
            quarantined_ambiguous=int(qc), redundant_duplicates_removed=int(rd),
            retained_recordings=remain,
            remaining_speakers=int(df[df.corpus == c].speaker_id.nunique()))
        if eligible and (qc+rd)/eligible > .05:
            warnings.append(f'{c}: {(qc+rd)/eligible:.1%} of eligible files excluded '
                            'by duplicates/quarantine. Check quality and report exclusion bias.')
        log.info('[AUDIT SUMMARY] %-8s scanned=%d, invalid=%d, skipped=%d, '
                 'benign_dropped=%d, conflict_quarantined=%d, retained=%d, speakers=%d',
                 c, scanned_files[c], integrity_summary[c]['rejected_audio'], excluded[c],
                 rd, qc, remain, integrity_summary[c]['remaining_speakers'])
    unsafe_groups = [g for g in groups if g['action'] == 'quarantine_all']
    for g in unsafe_groups[:20]:
        log.warning('[AUDIT QUARANTINE] %s: %s; %d files; %s',
                    g['group_id'], g['reason'], g['total_files'], g['recording_ids'][:180])
    if len(unsafe_groups) > 20:
        log.warning('[AUDIT] %d more unsafe groups logged in duplicate_groups.csv',
                    len(unsafe_groups)-20)
    if len(groups) > 30:
        log.warning('[AUDIT] %d duplicate groups found; see duplicate_groups.csv for ALL groups',
                    len(groups))
    viability_errors = list(scan_errors)
    selected = {}
    for c in CORPORA:
        sub = df[df.corpus == c]
        if sub.speaker_id.nunique() < 3:
            viability_errors.append(f'{c}: fewer than 3 speakers remaining ({sub.speaker_id.nunique()})')
        missing = [LABELS[k] for k in range(NC) if int((sub.emotion_id == k).sum()) < 1]
        if missing:
            viability_errors.append(f'{c}: absent emotion classes after quarantine: {missing}')
        if sub.speaker_id.nunique() >= 3 and not missing:
            try:
                selected[c] = choose_val_speakers(sub, a.val_speaker_fraction, a.seed)
                log.info('[SPEAKER VAL] %-8s: %s', c, selected[c])
            except (RuntimeError, ValueError) as exc:
                viability_errors.append(f'{c}: validation speaker selection failed: {exc}')

    fingerprint = hashlib.sha256(df[['recording_id', 'speaker_id', 'emotion_id',
          'rel_path', 'orig_frames', 'orig_sr', 'sha256', 'pcm_sha256']].to_csv(index=False).encode()).hexdigest()
    split_metadata = {
        'seed': a.seed, 'val_speaker_fraction': a.val_speaker_fraction,
        'validation_speakers': selected, 'labels': LABELS, 'audit_policy_version': AUDIT_POLICY_VERSION,
        'duplicate_audit_fingerprint': hashlib.sha256(json.dumps(
              group_details, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
              if not a.skip_wav_hash else 'HASH_AUDIT_DISABLED',
        'manifest_fingerprint': fingerprint}
    write_json(audit_dir/'audit_summary.json', {
        'protocol': PROTOCOL, 'audit_policy': AUDIT_POLICY_VERSION,
        'complete_scan': not scan_errors, 'audit_viable': not viability_errors,
        'six_emotions': LABELS, 'hash_checked': not a.skip_wav_hash,
        'data_root': str(root), 'by_corpus': integrity_summary,
        'bad_wav_files': len(rejects), 'n_duplicate_groups': len(groups),
        'n_ambiguous_groups': sum(g['action'] == 'quarantine_all' for g in groups),
        'ambiguous_audio_files_quarantined': len(quarantined),
        'benign_redundant_wavs_removed': len(redundant),
        'byte_identical_wavs_excluded': len(duplicates),
        'duplicate_policy': 'Retain one if same corpus/speaker/emotion and consistent recording ID; '
            'quarantine ALL members of every contradictory/cross-speaker/cross-corpus group; '
            'group by recording identity, WAV hash, or decoded mono PCM hash.',
        'raw_data_unchanged': True, 'test_corpora_are_never_used_for_checkpoint_selection': True,
        'manifest_fingerprint': fingerprint, 'scan_errors': scan_errors,
        'warnings': warnings, 'viability_errors': viability_errors,
        'counts': {c: {'recordings': int((df.corpus == c).sum()),
              'speakers': int(df[df.corpus == c].speaker_id.nunique())} for c in CORPORA}})
    review_lines = [
      '# Full four-corpus data quality audit', '',
      f'- Audit policy: `{AUDIT_POLICY_VERSION}`',
      f'- Input: `{root}`',
      f'- Total valid eligible before group checks: **{len(candidates)}**',
      f'- Retained: **{len(df)}**',
      f'- Rejected invalid audio/filenames: **{len(rejects)}**',
      f'- Safe redundant copies removed: **{len(redundant)}**',
      f'- Ambiguous recordings quarantined: **{len(quarantined)}**, in '
       f'**{sum(g["action"] == "quarantine_all" for g in groups)}** groups', '',
      '## Per corpus', '',
      '| Corpus | Scanned paths | Excluded labels | Invalid | Benign removed | Conflict quarantined | Retained |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for c in CORPORA:
        v = integrity_summary[c]
        review_lines.append(f'| {c} | {v["scanned_files"]} | {v["excluded_unshared_labels"]} | '
          f'{v["rejected_audio"]} | {v["redundant_duplicates_removed"]} | '
          f'{v["quarantined_ambiguous"]} | {v["retained_recordings"]} |')
    review_lines.extend(['', '## Integrity policy', '',
      'Exact repeat, same corpus/speaker/emotion: retain one. Conflicting emotion, '
      'speaker, corpus, or repeated recording identity with different audio: quarantine '
      '**the complete connected group**, never guess the correct label. Compare SHA-256 '
      'of WAV bytes and decoded mono PCM at the original sample rate. These checks do '
      'not detect near-duplicates, audio with different sample rates, or semantic '
      'annotation mistakes without repeated audio.', '',
      'Files: `full_scan_inventory.csv`, `rejected_audio.csv`, `duplicate_audio.csv`, '
      '`duplicate_groups.csv`, `quarantined_audio.csv`, '
      '`class_counts_before_after_dedup.csv`, `duplicate_group_details.json`.', '',
      '## Warnings', '', *(f'- {w}' for w in warnings)] )
    if not warnings:
        review_lines.append('- None triggered by the 5% duplicate-exclusion warning threshold.')
    review_lines.extend(['', '## Dataset viability', ''])
    review_lines.extend(f'- **BLOCKER**: {e}' for e in viability_errors)
    if not viability_errors:
        review_lines.append('- Pass: all six emotions and source-speaker validation remain feasible.')
    (audit_dir/'AUDIT_REVIEW.md').write_text('\n'.join(review_lines)+'\n', encoding='utf-8')

    if viability_errors:
        raise RuntimeError(f'Audit scanned all corpora and saved diagnostics, but found '
            f'{len(viability_errors)} dataset viability issue(s): {viability_errors}. '
            f'Read {audit_dir/"AUDIT_REVIEW.md"}.')
    immutable = audit_dir/'immutable_manifest_and_val_speakers.json'
    if immutable.exists() and json.loads(immutable.read_text()) != json.loads(json.dumps(split_metadata)):
        raise RuntimeError('The resolved dataset manifest/split changed relative to an existing run. '
                           f'Use a NEW --output-dir for this audit policy ({AUDIT_POLICY_VERSION}). '
                           f'Full diagnostics are available in {audit_dir}.')
    write_json(immutable, split_metadata)
    if all((df.corpus == c).any() for c in CORPORA):
        make_audit_plots(df, histogram, audit_dir)
    log.info('[AUDIT PASSED] all four corpora scanned | kept=%d, invalid=%d, '
             'safe_duplicates_removed=%d, ambiguous_quarantined=%d, groups=%d',
             len(df), len(rejects), len(redundant), len(quarantined), len(groups))
    if warnings:
        for warning in warnings:
            log.warning('[AUDIT RESEARCH CAUTION] %s', warning)
    return df, selected, split_metadata


def choose_val_speakers(frame, fraction, seed):
    """Deterministically pick 20% source speakers, maximizing class coverage.

    For small corpora, all six emotions in held-out speaker(s) are required.
    Labels from the eventual *test* corpus are not used here, as this function
    only determines candidate source partitions; no hyperparameters are tuned.
    """
    corpus = str(frame.corpus.iloc[0])
    speakers = sorted(frame.speaker_id.unique())
    n = max(1, min(len(speakers)-2, round(len(speakers)*fraction)))
    if n < 1:
        raise RuntimeError('Not enough speakers')
    by_speaker = {s: frame[frame.speaker_id == s].emotion_id.value_counts().reindex(range(NC), fill_value=0).to_numpy()
                  for s in speakers}
    best, best_rank = None, None
    rng = np.random.default_rng(stable_seed(seed, corpus, 'val_speakers'))
    # Candidate sets use source metadata only. Optimization favors class coverage
    # and stable class support, not predictive performance.
    for i in range(3000):
        ids = tuple(sorted(rng.choice(speakers, n, replace=False).tolist()))
        count = sum((by_speaker[s] for s in ids), np.zeros(NC, dtype=int))
        train_count = sum((by_speaker[s] for s in speakers if s not in ids), np.zeros(NC, dtype=int))
        if (train_count < 2).any() or (count == 0).any():
            continue
        rank = (int(count.min()), -float(np.std(count/(count.sum()+1e-12))), -i)
        if best_rank is None or rank > best_rank:
            best, best_rank = ids, rank
    if best is None:
        raise RuntimeError(f'Cannot select {n} validation speakers covering all six emotions in {corpus}')
    return list(best)


def make_audit_plots(df, histogram, out):
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(NC)
    for i, corpus in enumerate(CORPORA):
        ax.bar(x + (i - 1.5)*0.19, histogram.loc[corpus].values, 0.19, label=corpus)
    ax.set_xticks(x, LABELS)
    ax.set_ylabel('Original recordings')
    ax.set_title('Shared six-class distribution, before training/validation allocation')
    ax.legend(ncol=4)
    fig.tight_layout()
    fig.savefig(out/'class_counts_by_corpus.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.boxplot([df.loc[df.corpus == c, 'duration_s'].values for c in CORPORA], showfliers=False)
    ax.set_xticks(range(1, len(CORPORA)+1), list(CORPORA))
    ax.set_ylabel('Audio duration (seconds)')
    ax.set_title('Corpus duration distributions, original utterances')
    fig.tight_layout()
    fig.savefig(out/'audio_duration_boxplot.png', dpi=200)
    plt.close(fig)


def partition_experiment(df, val_map, heldout):
    source = df[df.corpus != heldout].copy()
    test = df[df.corpus == heldout].copy()
    val_speakers = set(s for c in CORPORA if c != heldout for s in val_map[c])
    val = source[source.speaker_id.isin(val_speakers)].copy()
    train = source[~source.speaker_id.isin(val_speakers)].copy()
    assert len(train)+len(val)+len(test) == len(df)
    assert not (set(train.recording_id) & set(val.recording_id))
    assert not (set(train.recording_id) & set(test.recording_id))
    assert not (set(val.recording_id) & set(test.recording_id))
    assert not (set(train.speaker_id) & set(val.speaker_id))
    assert set(test.corpus) == {heldout}
    assert set(train.corpus) == set(val.corpus) == set(CORPORA)-{heldout}
    for c in set(train.corpus):
        assert set(train[train.corpus == c].emotion_id) == set(range(NC))
        assert set(val[val.corpus == c].emotion_id) == set(range(NC))
    return train, val, test


class AudioStore:
    def __init__(self, cache_dir, enabled=True):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = enabled

    def get(self, row):
        cache_key = hashlib.sha256((row['recording_id']+'|'+str(row['sha256'])+'|'+str(row['orig_frames'])).encode()).hexdigest()[:24]
        path = self.cache_dir/(cache_key+'.npy')
        if self.enabled and path.exists():
            x = np.load(path, mmap_mode='r', allow_pickle=False)
            if x.ndim != 1 or len(x) < 160 or not np.isfinite(x).all():
                raise RuntimeError(f'Bad audio cache {path}')
            return x
        w, sr = sf.read(row['abs_path'], dtype='float32', always_2d=True)
        x = w.mean(axis=1, dtype=np.float32)
        if sr != SAMPLE_RATE:
            g = math.gcd(int(sr), SAMPLE_RATE)
            x = resample_poly(x, SAMPLE_RATE//g, int(sr)//g, window=('kaiser', 8.6)).astype(np.float32)
        if len(x) < 160 or not np.isfinite(x).all() or float(np.max(np.abs(x))) < 1e-7:
            raise ValueError(f'Invalid/near-silent audio: {row["abs_path"]}')
        # No loudness normalization/trim here: WavLM feature extractor performs
        # masked mean/variance normalization; pauses can be emotion-relevant.
        if self.enabled:
            temp = path.with_name(path.name+f'.{os.getpid()}.tmp')
            with open(temp, 'wb') as f:
                np.save(f, x, allow_pickle=False)
            os.replace(temp, path)
            return np.load(path, mmap_mode='r', allow_pickle=False)
        return x


def expected_len(row):
    n, sr = int(row['orig_frames']), int(row['orig_sr'])
    up, down = SAMPLE_RATE//math.gcd(sr, SAMPLE_RATE), sr//math.gcd(sr, SAMPLE_RATE)
    return (n*up+down-1)//down


def eval_slices(n, window, hop_fraction=.5, max_windows=5):
    if n <= window:
        return [(0, n)]
    last = n-window
    hop = max(1, round(window*hop_fraction))
    count = min(max_windows, math.ceil(last/hop)+1)
    positions = np.linspace(0, last, count).round().astype(int)
    return [(int(s), int(s+window)) for s in np.unique(positions)]


def random_train_slice(row, x, window, epoch, seed):
    rng = np.random.default_rng(stable_seed(seed, epoch, row['recording_id'], 'crop'))
    last = len(x)-window
    if last <= 0:
        clip = np.asarray(x, dtype=np.float32).copy()
    else:
        # Rotate the utterance's 5 regions across epochs; mild 300-ms jitter.
        region = (epoch+stable_seed(row['recording_id'])%5)%5
        anchor = round(last*region/4)
        start = int(np.clip(anchor+round(rng.uniform(-.30, .30)*SAMPLE_RATE), 0, last))
        clip = np.asarray(x[start:start+window], dtype=np.float32).copy()
    if rng.random() < .15:
        rms = float(np.sqrt(np.mean(clip.astype(np.float64)**2)))
        if rms > 1e-5:
            noise = rng.normal(0, rms/(10**(rng.uniform(23, 36)/20)), len(clip)).astype('float32')
            clip = np.clip(clip + noise, -1, 1)
    return clip


class TrainDataset:
    def __init__(self, df, store, window, seed):
        self.rows = df.sort_values('recording_id').to_dict('records')
        self.store = store
        self.window = window
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        r = self.rows[index]
        x = self.store.get(r)
        return {'wave': random_train_slice(r, x, self.window, self.epoch, self.seed),
                'y': r['emotion_id'], 'recording_id': r['recording_id'],
                'corpus': r['corpus'], 'speaker_id': r['speaker_id']}


class EvalDataset:
    def __init__(self, df, store, window, hop_fraction, max_windows):
        self.rows = df.sort_values('recording_id').to_dict('records')
        self.store = store
        self.items = [(i, start, stop)
                      for i, r in enumerate(self.rows)
                      for start, stop in eval_slices(expected_len(r), window, hop_fraction, max_windows)]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        i, start, stop = self.items[index]
        row = self.rows[i]
        wave = self.store.get(row)
        if abs(len(wave)-expected_len(row)) > 1:
            raise RuntimeError(f'Audio resampling length mismatch: {row["recording_id"]}')
        x = np.asarray(wave[start:min(stop, len(wave))], dtype=np.float32).copy()
        if len(x) < 160:
            raise RuntimeError('Too short eval window')
        return {'wave': x, 'y': row['emotion_id'], 'recording_id': row['recording_id'],
                'corpus': row['corpus'], 'speaker_id': row['speaker_id']}


class CorpusCrossSpeakerPairSampler:
    """Equal class exposure and equal corpus-pair exposure, without test examples.

    Each positive same-emotion pair comes from DIFFERENT source corpora, so the
    contrastive objective is not rewarded for merely detecting a common speaker
    or recording environment. Speakers are selected uniformly within corpus/class.
    The sampler draws with replacement, which is required for small EMOVO/EmoDB.
    """
    def __init__(self, ds, batch_size, n_batches, seed):
        self.ds, self.batch_size, self.n_batches, self.seed = ds, batch_size, n_batches, seed
        self.epoch = 0
        self.by_class_corpus_speaker = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(ds.rows):
            key = (int(row['emotion_id']), row['corpus'])
            self.by_class_corpus_speaker[key][row['speaker_id']].append(i)
        self.corpora = tuple(sorted({r['corpus'] for r in ds.rows}))
        if len(self.corpora) != 3:
            raise RuntimeError('Exactly three source corpora required')
        for cls in range(NC):
            for c in self.corpora:
                if not self.by_class_corpus_speaker[(cls, c)]:
                    raise RuntimeError(f'Missing training examples in class {cls}, corpus {c}')
        if batch_size not in (2, 6, 12) or n_batches % (12//batch_size):
            raise ValueError('Batch size must be 2, 6, 12 and must cover complete six-emotion cycles')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, 'corpus_pair_sampler'))
        pairs = [(0, 1), (1, 2), (0, 2)]
        for batch_no in range(self.n_batches):
            # Every complete cycle covers all six classes once, with one
            # cross-corpus positive pair per class; works for 2/6/12 items.
            pair_count = self.batch_size//2
            batches_per_cycle = NC//pair_count
            group, position = divmod(batch_no, batches_per_cycle)
            if position == 0:
                order = rng.permutation(NC)
            classes = list(map(int, order[position*pair_count:(position+1)*pair_count]))
            batch = []
            for cls in classes:
                # Rotating source pair distributes source membership evenly.
                ca, cb = pairs[(group+cls+self.epoch)%len(pairs)]
                for cidx in (ca, cb):
                    corpus = self.corpora[cidx]
                    speakers = sorted(self.by_class_corpus_speaker[(cls, corpus)])
                    speaker = speakers[int(rng.integers(len(speakers)))]
                    options = self.by_class_corpus_speaker[(cls, corpus)][speaker]
                    batch.append(int(options[int(rng.integers(len(options)))]))
            rng.shuffle(batch)
            assert len(batch) == self.batch_size
            yield batch


def collate_factory(extractor, window):
    import torch
    def collate(items):
        features = extractor([x['wave'] for x in items], sampling_rate=SAMPLE_RATE,
                             padding='max_length', max_length=window, truncation=True,
                             return_tensors='pt', return_attention_mask=True)
        mask = features['attention_mask'].long()
        if (mask.sum(-1) < 160).any():
            raise RuntimeError('Invalid audio attention mask')
        return {'input_values': features['input_values'], 'attention_mask': mask,
                'y': torch.tensor([x['y'] for x in items], dtype=torch.long),
                'recording_id': [x['recording_id'] for x in items],
                'speaker_id': [x['speaker_id'] for x in items],
                'corpus': [x['corpus'] for x in items]}
    return collate


def plot_heatmap(mat, xlabels, ylabels, path, title, percentages=False):
    mat = np.asarray(mat)
    fig, ax = plt.subplots(figsize=(max(7, len(xlabels)*1.25), max(5, len(ylabels)*.7)))
    im = ax.imshow(mat, cmap='Blues', vmin=0, vmax=1 if percentages else None, aspect='auto')
    for i in range(len(ylabels)):
        for j in range(len(xlabels)):
            val = float(mat[i, j])
            ax.text(j, i, f'{val:.2f}' if percentages else str(int(val)),
                    ha='center', va='center', fontsize=9,
                    color='white' if im.norm(val) > .7 else 'black')
    ax.set_xticks(range(len(xlabels)), xlabels, rotation=25, ha='right')
    ax.set_yticks(range(len(ylabels)), ylabels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=.8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches='tight')
    plt.close(fig)


def classification_metrics(df, n_bootstrap=5000, seed=42):
    y = df.true_class.to_numpy(dtype=int)
    yp = df.pred_class.to_numpy(dtype=int)
    prob = df[[f'p_{s}' for s in SLUGS]].to_numpy(dtype=float)
    if not len(y) or not np.isfinite(prob).all() or (np.abs(prob.sum(1)-1) > 1e-4).any():
        raise RuntimeError('Invalid eval probabilities')
    precision, recall, f1, support = precision_recall_fscore_support(y, yp, labels=range(NC), zero_division=0)
    metrics = {'n_recordings': len(y), 'n_speakers': df.speaker_id.nunique(),
               'accuracy': float(accuracy_score(y, yp)),
               'uar': float(np.mean(recall)),
               'macro_f1': float(np.mean(f1)),
               'weighted_f1': float(np.average(f1, weights=support)),
               'cross_entropy': float(log_loss(y, prob, labels=list(range(NC))))}
    try:
        metrics['auroc_macro_ovr'] = float(roc_auc_score(y, prob, labels=list(range(NC)),
                                                         multi_class='ovr', average='macro'))
    except ValueError:
        metrics['auroc_macro_ovr'] = None
    conf = prob.max(1)
    bins = np.minimum((conf*15).astype(int), 14)
    metrics['ece_15_bins'] = float(sum((bins == b).mean()*abs((yp[bins == b] == y[bins == b]).mean()-conf[bins == b].mean())
                                        for b in range(15) if (bins == b).any()))
    cm = confusion_matrix(y, yp, labels=list(range(NC)))
    class_df = pd.DataFrame({'class_id': range(NC), 'emotion': LABELS, 'precision': precision,
                             'recall': recall, 'f1': f1, 'support': support})
    speakers = []
    for speaker, g in df.groupby('speaker_id', sort=True):
        gy = g.true_class.to_numpy(int)
        gp = g.pred_class.to_numpy(int)
        _, gr, gf, _ = precision_recall_fscore_support(gy, gp, labels=range(NC), zero_division=0)
        speakers.append({'speaker_id': speaker, 'corpus': g.corpus.iloc[0], 'n_recordings': len(g),
                         'accuracy': float(np.mean(gy == gp)), 'uar': float(np.mean(gr)),
                         'macro_f1': float(np.mean(gf))})
    spdf = pd.DataFrame(speakers)
    metrics['mean_speaker_accuracy'] = float(spdf.accuracy.mean())
    metrics['mean_speaker_uar'] = float(spdf.uar.mean())
    metrics['bootstrap_unit'] = 'speaker (cluster bootstrap)'
    # Whole speakers are resampled to preserve within-speaker correlation.
    rng = np.random.default_rng(stable_seed(seed, 'bootstrap', '|'.join(sorted(df.corpus.unique()))))
    buckets = [g.index.to_numpy() for _, g in df.reset_index(drop=True).groupby('speaker_id', sort=True)]
    # df indices should be 0..N-1; scores are computed from positional arrays.
    scores = []
    for _ in range(n_bootstrap):
        choices = rng.integers(len(buckets), size=len(buckets))
        indices = np.concatenate([buckets[i] for i in choices])
        yt, pt = y[indices], yp[indices]
        r = precision_recall_fscore_support(yt, pt, labels=range(NC), zero_division=0)[1]
        scores.append((float(np.mean(yt == pt)), float(np.mean(r))))
    ci = np.percentile(scores, [2.5, 97.5], axis=0)
    metrics['accuracy_95ci_speaker_bootstrap'] = [float(ci[0, 0]), float(ci[1, 0])]
    metrics['uar_95ci_speaker_bootstrap'] = [float(ci[0, 1]), float(ci[1, 1])]
    return metrics, spdf, class_df, cm


def plot_embedding_pca(embedding_matrix, preds, output, title):
    """Descriptive post-hoc 2-D visualization only; NOT used for model selection."""
    from sklearn.decomposition import PCA
    if embedding_matrix.shape != (len(preds), 512):
        raise RuntimeError('Incorrect WavLM/DPTP embedding matrix shape')
    pts = PCA(n_components=2, svd_solver='randomized', random_state=42).fit_transform(embedding_matrix)
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, label in enumerate(LABELS):
        choice = preds.true_class.to_numpy(int) == i
        ax.scatter(pts[choice, 0], pts[choice, 1], s=15, alpha=.55, label=label)
    ax.set_xlabel('First PCA component')
    ax.set_ylabel('Second PCA component')
    ax.set_title(title+' (descriptive; not a performance measure)')
    ax.legend(markerscale=1.5)
    fig.tight_layout()
    fig.savefig(output, dpi=200)
    plt.close(fig)


def plot_evaluation(pred, metrics, speaker, classes, cm, folder, title):
    folder.mkdir(parents=True, exist_ok=True)
    plot_heatmap(cm, LABELS, LABELS, folder/'confusion_counts.png', f'{title}: confusion (counts)')
    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    plot_heatmap(norm, LABELS, LABELS, folder/'confusion_normalized.png',
                 f'{title}: row-normalized confusion', percentages=True)
    xx = np.arange(NC)
    fig, ax = plt.subplots(figsize=(10, 5))
    for j, (key, name) in enumerate([('precision', 'Precision'), ('recall', 'Recall'), ('f1', 'F1')]):
        ax.bar(xx+(j-1)*.25, 100*classes[key].to_numpy(), .25, label=name)
    ax.set_xticks(xx, LABELS)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Percent')
    ax.set_title(f'{title}: class-wise scores')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(folder/'per_class_precision_recall_f1.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(max(9, .26*len(speaker)), 4.8))
    ax.bar(np.arange(len(speaker)), speaker.accuracy.to_numpy()*100)
    ax.set_xticks(np.arange(len(speaker)), [s.rsplit(':', 1)[-1] for s in speaker.speaker_id], rotation=75)
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%)')
    ax.set_xlabel('Speaker ID')
    ax.set_title(f'{title}: individual speakers (n={len(speaker)})')
    fig.tight_layout()
    fig.savefig(folder/'per_speaker_accuracy.png', dpi=200)
    plt.close(fig)
    y = pred.true_class.to_numpy(int)
    prob = pred[[f'p_{s}' for s in SLUGS]].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(LABELS):
        binary = (y == i).astype(int)
        if len(np.unique(binary)) < 2:
            continue
        fpr, tpr, _ = roc_curve(binary, prob[:, i])
        ap = roc_auc_score(binary, prob[:, i])
        ax.plot(fpr, tpr, label=f'{label} (AUC={ap:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1)
    ax.set_xlabel('False positive rate')
    ax.set_ylabel('True positive rate')
    ax.set_title(f'{title}: one-vs-rest ROC')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(folder/'roc_ovr.png', dpi=200)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, label in enumerate(LABELS):
        binary = (y == i).astype(int)
        if binary.sum() == 0:
            continue
        p, r, _ = precision_recall_curve(binary, prob[:, i])
        ap = average_precision_score(binary, prob[:, i])
        ax.plot(r, p, label=f'{label} (AP={ap:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'{title}: one-vs-rest precision-recall')
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(folder/'pr_ovr.png', dpi=200)
    plt.close(fig)
    confidence = prob.max(axis=1)
    good = (pred.true_class == pred.pred_class).to_numpy()
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(confidence[good], bins=np.linspace(0, 1, 21), alpha=.7, label='Correct')
    if (~good).any():
        ax.hist(confidence[~good], bins=np.linspace(0, 1, 21), alpha=.7, label='Incorrect')
    ax.set_xlabel('Maximum softmax probability')
    ax.set_ylabel('Recordings')
    ax.set_title(f'{title}: prediction confidence')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'prediction_confidence.png', dpi=200)
    plt.close(fig)
    # Reliability diagram, useful for documenting out-of-domain calibration.
    bins = np.minimum((confidence*15).astype(int), 14)
    observed_x, observed_y, counts = [], [], []
    for b in range(15):
        sel = bins == b
        if sel.any():
            observed_x.append(float(confidence[sel].mean()))
            observed_y.append(float(good[sel].mean()))
            counts.append(int(sel.sum()))
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ax.plot([0, 1], [0, 1], '--', color='gray', label='Perfect calibration')
    ax.scatter(observed_x, observed_y, s=np.maximum(20, np.sqrt(counts)*22), label='Confidence bins')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel('Mean confidence')
    ax.set_ylabel('Empirical accuracy')
    ax.set_title(f'{title}: reliability (ECE={metrics["ece_15_bins"]:.3f})')
    ax.legend()
    fig.tight_layout()
    fig.savefig(folder/'reliability_diagram.png', dpi=200)
    plt.close(fig)


def export_scores(preds, output, title, n_bootstrap, seed):
    preds = preds.sort_values('recording_id').reset_index(drop=True)
    metrics, speaker, class_df, cm = classification_metrics(preds, n_bootstrap, seed)
    output.mkdir(parents=True, exist_ok=True)
    preds.to_csv(output/'recording_predictions.csv', index=False)
    speaker.to_csv(output/'per_speaker_metrics.csv', index=False)
    class_df.to_csv(output/'per_class_metrics.csv', index=False)
    pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(output/'confusion_counts.csv')
    pd.DataFrame(cm/np.maximum(cm.sum(1, keepdims=True), 1), index=LABELS, columns=LABELS).to_csv(
        output/'confusion_normalized.csv')
    write_json(output/'metrics.json', metrics)
    plot_evaluation(preds, metrics, speaker, class_df, cm, output, title)
    return metrics


# The following model classes deliberately retain the supplied PME4 architecture;
# only the final classifier output dimensionality is changed from 4 to 6.
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler



def verify_numpy_torch_compatibility():
    """Fail before a long audit if PyTorch was built against an incompatible NumPy ABI."""
    import numpy as _np
    try:
        candidate = _np.asarray([0.25, -0.5], dtype=_np.float32)
        tensor = torch.from_numpy(candidate)
        check = tensor.cpu().numpy()
        if check.shape != (2,) or not _np.allclose(check, candidate):
            raise RuntimeError('NumPy/Torch conversion self-test returned incorrect data')
    except Exception as exc:
        raise RuntimeError(
            'INCOMPATIBLE NUMPY/TORCH ENVIRONMENT: PyTorch cannot convert NumPy arrays. '
            'With your NVIDIA PyTorch build, activate the intended venv and install '
            'numpy==1.26.4 in that venv, then verify torch.from_numpy works. '
            f'Python={sys.executable}; NumPy={_np.__version__}; '
            f'NumPy path={_np.__file__}; PyTorch={torch.__version__}. '
            'The Hugging Face padding exception is only a downstream symptom.'
        ) from exc


def recover_crashed_final_test(output, lock, log, heldout, checkpoint_sha256):
    """Recover only a test that has no persisted final predictions/results.

    This does not permit retesting a completed or partially exported evaluation.
    The interrupted attempt is retained as an audit record rather than deleted.
    """
    output = Path(output)
    if (output/'final_test_receipt.json').exists():
        raise RuntimeError('Completed held-out test receipt exists; recovery forbidden')
    result_folder = output/'test'
    existing = sorted(str(x.relative_to(output)) for x in result_folder.rglob('*') if x.is_file()) if result_folder.exists() else []
    if existing:
        raise RuntimeError('Refusing held-out test retry because test output files already exist: '
                           +repr(existing[:15])+'. Review manually; do not retest a published result.')
    history_path = output/'interrupted_test_recovery.json'
    past = json.loads(history_path.read_text()) if history_path.exists() else []
    # Preserve the original lock with a stable, non-overwriting suffix.
    dest = output/('final_test_interrupted_attempt_%03d.lock' % (len(past)+1))
    if dest.exists():
        raise RuntimeError('Recovery archive already exists and was not journaled; manual inspection required')
    original_text = lock.read_text(encoding='utf8')
    os.replace(lock, dest)
    entry = {'recovered_at_utc': datetime.now(timezone.utc).isoformat(),
             'heldout_corpus': heldout, 'checkpoint_sha256': checkpoint_sha256,
             'archived_interrupted_lock': dest.name,
             'original_lock_content': original_text,
             'reason': 'Explicit user-initiated recovery after evaluation crash with NO persisted test artifacts',
             'warning': 'Recover only infrastructure failure; do not use target data to tune the model.'}
    past.append(entry)
    write_json(history_path, past)
    log.warning('[%s] RECOVERED interrupted test attempt without results. Archived %s. '
                'Final test remains untuned and the recovery is journaled.', heldout, dest.name)


class GroupedHierarchicalFusion(nn.Module):
    """Four groups x six WavLM transformer layers, learned within-group fusion."""
    def __init__(self, hidden_size=1024):
        super().__init__()
        self.layer_logits = nn.Parameter(torch.zeros(4, 6))
        self.group_gate = nn.Linear(hidden_size, 1)
        nn.init.zeros_(self.group_gate.weight)
        nn.init.zeros_(self.group_gate.bias)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, mask):
        if len(hidden_states) != 24:
            raise RuntimeError(f'Expected WavLM transformer layers 1..24, got {len(hidden_states)}')
        mask = mask.bool()
        if (~mask.any(dim=1)).any():
            raise RuntimeError('All-padding audio in GHF')
        valid = mask.float().sum(dim=1).clamp(min=1).unsqueeze(-1)
        weights = F.softmax(self.layer_logits.float(), dim=-1)
        groups, gates = [], []
        for gi in range(4):
            h = sum(hidden_states[gi*6+j]*weights[gi, j].to(hidden_states[0].dtype)
                    for j in range(6))
            pooled = (h.float()*mask.unsqueeze(-1)).sum(dim=1)/valid
            gates.append(torch.sigmoid(self.group_gate(pooled)))
            groups.append(h)
        gate_mat = torch.cat(gates, dim=-1)
        denom = gate_mat.sum(-1, keepdim=True).clamp(min=1e-6)
        x = sum(groups[g]*gate_mat[:, g, None, None].to(groups[g].dtype) for g in range(4))
        x = x/denom[:, None, :].to(x.dtype)
        return self.norm(x), weights, gate_mat


class DualPathTemporalPooling(nn.Module):
    """Learnable query attention + masked mean/std pooling -> 512-D embedding."""
    def __init__(self, dim=1024, heads=8):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, dim)*.02)
        self.attention = nn.MultiheadAttention(dim, heads, dropout=.10, batch_first=True)
        self.stats_proj = nn.Linear(2*dim, dim)
        self.stats_norm = nn.LayerNorm(dim)
        self.bottleneck = nn.Linear(dim, 512)
        self.output_norm = nn.LayerNorm(512)

    def forward(self, x, mask):
        b, t, h = x.shape
        if not mask.bool().any(dim=1).all():
            raise RuntimeError('All-padding audio in DPTP')
        q = self.query.expand(b, -1, -1).to(x.dtype)
        attn, _ = self.attention(q, x, x, key_padding_mask=~mask.bool(), need_weights=False)
        valid = mask.float().unsqueeze(-1)
        count = valid.sum(dim=1).clamp(min=1)
        xf = x.float()
        mu = (xf*valid).sum(dim=1)/count
        var = (((xf-mu.unsqueeze(1))**2)*valid).sum(dim=1)/count
        std = (var.clamp(min=0)+1e-5).sqrt()
        stats = F.silu(self.stats_norm(self.stats_proj(torch.cat([mu, std], dim=-1).to(x.dtype))))
        z = F.silu(self.output_norm(self.bottleneck(attn[:, 0, :]+stats)))
        return z


def conv_frame_lengths(input_len, kernels, strides):
    lengths = input_len.long()
    for kernel, stride in zip(kernels, strides):
        lengths = torch.div(lengths-int(kernel), int(stride), rounding_mode='floor')+1
    return lengths


class ArchitectureA(nn.Module):
    def __init__(self, wavlm):
        super().__init__()
        self.wavlm = wavlm
        self.ghf = GroupedHierarchicalFusion(1024)
        self.dptp = DualPathTemporalPooling(1024, heads=8)
        self.emotion_head = nn.Sequential(nn.Linear(512, 256), nn.GELU(),
                                          nn.Dropout(.30), nn.Linear(256, NC))
        self.projector = nn.Sequential(nn.Linear(512, 256), nn.GELU(), nn.Linear(256, 128))
        self.stage = 'frozen'

    def set_stage(self, stage):
        if stage not in ('frozen', 'finetune'):
            raise ValueError(stage)
        self.stage = stage
        self.wavlm.requires_grad_(False)
        if stage == 'finetune':
            for layer in self.wavlm.encoder.layers[18:24]:
                layer.requires_grad_(True)
        else:
            self.wavlm.eval()

    def train(self, mode=True):
        super().train(mode)
        if mode and self.stage == 'frozen':
            self.wavlm.eval()
        return self

    def forward(self, input_values, attention_mask, need_projection=False):
        ctx = torch.no_grad() if self.stage == 'frozen' else contextlib.nullcontext()
        with ctx:
            output = self.wavlm(input_values, attention_mask=attention_mask,
                                output_hidden_states=True, return_dict=True)
        hidden = output.hidden_states
        if len(hidden) != 25:
            raise RuntimeError('Expected 1 embedding + 24 WavLM hidden state tensors')
        frame_len = hidden[-1].shape[1]
        lengths = conv_frame_lengths(attention_mask.sum(-1),
                                     self.wavlm.config.conv_kernel,
                                     self.wavlm.config.conv_stride)
        if (lengths <= 0).any() or (lengths > frame_len).any():
            raise RuntimeError('WavLM feature frame length mismatch')
        frame_mask = torch.arange(frame_len, device=input_values.device)[None, :] < lengths[:, None]
        feat, layer_weights, group_gates = self.ghf(hidden[1:25], frame_mask)
        z = self.dptp(feat, frame_mask)
        logits = self.emotion_head(z)
        projection = F.normalize(self.projector(z).float(), dim=-1, eps=1e-8) if need_projection else None
        return {'logits': logits, 'embedding': z, 'projection': projection,
                'layer_weights': layer_weights, 'group_gates': group_gates}


def supcon_loss(projections, targets, temperature=.07):
    z = F.normalize(projections.float(), dim=-1, eps=1e-8)
    targets = targets.flatten()
    sim = z@z.T/temperature
    diag = torch.eye(len(targets), dtype=torch.bool, device=targets.device)
    positive = (targets[:, None] == targets[None, :]) & ~diag
    if not positive.any(dim=1).all():
        raise RuntimeError('SupCon missing a same-class positive example')
    sim = sim.masked_fill(diag, float('-inf'))
    lp = sim-torch.logsumexp(sim, dim=1, keepdim=True)
    loss = -torch.where(positive, lp, torch.zeros_like(lp)).sum(1)/positive.sum(1)
    if not torch.isfinite(loss).all():
        raise FloatingPointError('Non-finite SupCon loss')
    return loss.mean()


class SupervisedContrastiveQueue:
    """CPU-backed, detached memory bank for genuine SupCon negatives at batch size 2.

    Current two samples must share an emotion and come from distinct source corpora.
    The queue supplies other-class negatives and optional same-class positives.
    Queued examples are stop-gradient keys, NOT end-to-end 24-example SupCon.
    """
    def __init__(self, capacity=144):
        self.capacity = int(capacity)
        self.keys = deque(maxlen=self.capacity)
        self.labels = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.keys)

    def loss(self, projections, targets, temperature=0.07):
        if projections.ndim != 2 or len(projections) != 2 or targets.shape != (2,):
            raise ValueError('Queue SupCon requires an exact 2-example positive pair')
        if int(targets[0]) != int(targets[1]):
            raise ValueError('Microbatch 2 must be a positive pair')
        z = F.normalize(projections.float(), dim=-1, eps=1e-8)
        if not self.keys:
            return z.sum()*0.0  # First pair: no negative key exists yet.
        old = torch.stack(list(self.keys)).to(device=z.device, dtype=z.dtype)
        old = F.normalize(old, dim=-1, eps=1e-8)
        old_labels = torch.tensor(list(self.labels), dtype=targets.dtype, device=z.device)
        all_keys = torch.cat([z, old], dim=0)
        all_labels = torch.cat([targets, old_labels])
        similarity = z @ all_keys.T / temperature
        self_mask = torch.zeros_like(similarity, dtype=torch.bool)
        self_mask[0, 0] = True
        self_mask[1, 1] = True
        positives = targets[:, None].eq(all_labels[None, :]) & ~self_mask
        negatives = targets[:, None].ne(all_labels[None, :])
        if not positives.any(dim=1).all():
            raise RuntimeError('Positive pair missing in queue SupCon')
        if not negatives.any():
            return z.sum()*0.0  # No negatives yet: no meaningful contrastive loss.
        similarity = similarity.masked_fill(self_mask, float('-inf'))
        logp = similarity - torch.logsumexp(similarity, dim=1, keepdim=True)
        loss = -(torch.where(positives, logp, torch.zeros_like(logp)).sum(dim=1)
                 / positives.sum(dim=1)).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite memory-bank SupCon loss')
        return loss

    @torch.no_grad()
    def enqueue(self, projections, targets):
        # Store on HOST RAM: zero persistent VRAM consumption from the queue.
        for z, label in zip(projections.detach().float().cpu(), targets.detach().cpu()):
            self.keys.append(z.clone())
            self.labels.append(int(label))


def configure_cuda_budget(max_gpu_gib, overhead_gib, log):
    """Fail closed at the caching-allocator cap; NOT an NVML/MIG total-VRAM quota.

    Non-PyTorch CUDA allocations and the driver context are excluded, hence the
    2 GiB default headroom. An actual hard physical limit needs GPU partitioning
    (such as appropriately configured MIG), not just PyTorch memory_fraction.
    """
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    device = torch.device('cuda:0')
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    budget_bytes = int(max_gpu_gib*2**30)
    allocator_bytes = int((max_gpu_gib-overhead_gib)*2**30)
    if budget_bytes > total_bytes:
        raise RuntimeError(f'{max_gpu_gib:.2f} GiB budget exceeds visible GPU memory {total_bytes/2**30:.2f} GiB')
    # Must be done BEFORE any tensor/model is allocated on CUDA.
    torch.cuda.set_per_process_memory_fraction(allocator_bytes/total_bytes, device=device)
    log.info('[VRAM BUDGET] Target <= %.2f GiB total; PyTorch caching allocator capped '
             'at %.2f GiB; %.2f GiB reserved as CUDA/non-PyTorch headroom. '
             'Physical process memory cannot be strictly guaranteed without MIG.',
             max_gpu_gib, allocator_bytes/2**30, overhead_gib)
    return allocator_bytes


def gpu_memory_snapshot(device):
    """PyTorch allocated/reserved plus optional nvidia-smi process memory."""
    torch.cuda.synchronize(device)
    out = {'allocated_gib': torch.cuda.memory_allocated(device)/2**30,
           'reserved_gib': torch.cuda.memory_reserved(device)/2**30,
           'peak_allocated_gib': torch.cuda.max_memory_allocated(device)/2**30,
           'peak_reserved_gib': torch.cuda.max_memory_reserved(device)/2**30,
           'process_nvidia_smi_gib': None}
    try:
        data = subprocess.check_output(
            ['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
             '--format=csv,noheader,nounits'], stderr=subprocess.DEVNULL, timeout=4,
            text=True)
        entries = [line.split(',') for line in data.splitlines()]
        memory_mib = sum(float(parts[1].strip()) for parts in entries
                         if len(parts) >= 2 and parts[0].strip() == str(os.getpid()))
        if memory_mib > 0:
            out['process_nvidia_smi_gib'] = memory_mib/1024.
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError, IndexError, OSError):
        pass
    return out


def quick_metrics(pred):
    y = pred.true_class.to_numpy(int)
    yp = pred.pred_class.to_numpy(int)
    prob = pred[[f'p_{s}' for s in SLUGS]].to_numpy(float)
    _, r, f, support = precision_recall_fscore_support(y, yp, labels=range(NC), zero_division=0)
    return {'accuracy': float(np.mean(y == yp)), 'uar': float(np.mean(r)),
            'macro_f1': float(np.mean(f)),
            'cross_entropy': float(log_loss(y, prob, labels=list(range(NC)))),
            'n_recordings': int(len(y))}


def plot_training(history, output, heldout):
    if not history:
        return
    df = pd.DataFrame(history)
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs[0, 0].plot(df.epoch, df.train_loss, label='train objective')
    axs[0, 0].plot(df.epoch, df.val_loss, label='val cross-entropy')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 1].plot(df.epoch, df.train_accuracy*100, label='train crop accuracy')
    axs[0, 1].plot(df.epoch, df.val_macro_corpus_uar*100, label='val mean corpus UAR')
    axs[0, 1].plot(df.epoch, df.val_macro_corpus_f1*100, label='val mean corpus macro-F1')
    axs[0, 1].set_ylim(0, 105)
    axs[0, 1].set_ylabel('Percent')
    axs[0, 1].legend()
    for corpus in [c for c in CORPORA if c != heldout]:
        axs[1, 0].plot(df.epoch, 100*df[f'val_uar_{corpus}'], label=corpus)
    axs[1, 0].set_ylabel('Source validation UAR (%)')
    axs[1, 0].legend()
    axs[1, 1].plot(df.epoch, df.peak_gpu_gib, label='peak GPU GiB')
    axs[1, 1].set_ylabel('GPU GiB')
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.axvline(df.loc[df.stage == 'finetune', 'epoch'].min()-.5,
                   color='gray', linestyle='--', linewidth=1)
    fig.suptitle(f'Architecture A LOCO — held-out {heldout.upper()} (validation excludes target)')
    fig.tight_layout()
    fig.savefig(output/'learning_curves.png', dpi=200)
    plt.close(fig)


def atomic_torch_save(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    torch.save(payload, temp)
    os.replace(temp, path)


class Experiment:
    def __init__(self, a, all_df, val_map, immutable, heldout, log):
        self.a, self.heldout, self.log = a, heldout, log
        self.train_df, self.val_df, self.test_df = partition_experiment(all_df, val_map, heldout)
        self.output = a.output_dir/('test_'+heldout.replace('-', '_'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.store = AudioStore(a.output_dir/'shared_16khz_audio_cache', a.cache_audio)
        self.window = round(a.window_seconds*SAMPLE_RATE)
        self.device = torch.device('cuda:0')
        self.split_fingerprint = hashlib.sha256((immutable['manifest_fingerprint']+'|'+heldout+'|'+
            self.train_df.recording_id.sort_values().to_csv(index=False)+'|'+
            self.val_df.recording_id.sort_values().to_csv(index=False)).encode()).hexdigest()
        self.config = {'protocol': PROTOCOL, 'held_out_corpus': heldout,
                       'training_corpora': [c for c in CORPORA if c != heldout],
                       'source_val_only': True, 'heldout_is_entire_corpus': True,
                       'six_class_ontology': LABELS,
                       'pretrained': PRETRAINED_MODEL, 'checkpoint_revision': a.checkpoint_revision,
                       'model_architecture': 'WavLM-Large [24x1024] -> GHF 4x6 -> DPTP [8-head attention + stats] -> 512 -> 256 -> 6',
                       'contrastive_projection': '512 -> 256 -> 128 normalized',
                       'learning_stages': f'{a.frozen_epochs} frozen + up to {a.max_epochs-a.frozen_epochs} top-6-layer finetuning',
                       'audio': {'resample_hz': SAMPLE_RATE, 'crop_seconds': a.window_seconds,
                                 'window_samples': self.window, 'train_crop': '5-region cyclic with +/-300ms jitter',
                                 'eval_sliding_hop_fraction': a.eval_hop_fraction,
                                 'max_eval_windows': a.max_eval_windows,
                                 'amplitude': 'no independent peak/loudness normalization; HF masked standardization',
                                 'silence_trimming': 'none; retain emotion-related pauses'},
                       'augmentation': '15% low-level Gaussian noise (23-36 dB SNR), training only',
                       'batch': {'micro': a.batch_size, 'grad_accum': a.grad_accum,
                                 'frozen_micro': a.frozen_batch_size,
                                 'frozen_grad_accum': a.frozen_grad_accum,
                                 'finetune_micro': a.batch_size,
                                 'finetune_grad_accum': a.grad_accum,
                                 'effective': a.batch_size*a.grad_accum,
                                 'updates_per_virtual_epoch': a.updates_per_epoch,
                                 'balanced_classes': True, 'cross_corpus_positive_pairs': True,
                                 'speaker_uniform_within_corpus_class': True},
                       'gpu_budget': {'target_gib': a.max_gpu_gib,
                                      'pytorch_allocator_gib': a.max_gpu_gib-a.cuda_overhead_gib,
                                      'non_pytorch_headroom_gib': a.cuda_overhead_gib},
                       'contrastive_queue_capacity': a.contrastive_queue_size,
                       'objective': 'CE(label_smoothing=0.05) + annealed 0.05 SupCon(tau=0.07); '
                                    'two-item batches use CPU queued stop-grad keys for negatives',
                       'checkpoint_selection': 'macro-average of source-corpus validation UAR; source speakers disjoint',
                       'early_stopping': {'patience': a.patience, 'min_delta': a.min_delta},
                       'optimizer_schedule': {'head_lr_frozen': 3e-4, 'head_lr_finetune': 1e-4,
                           'top6_lr': 5e-6, 'layerwise_decay': 0.8, 'weight_decay': 0.01,
                           'warmup_fraction_per_stage': 0.10, 'cosine_decay_floor': 0.05,
                           'max_epochs': a.max_epochs},
                       'seed': a.seed, 'source_validation_speakers': {
                           c: val_map[c] for c in CORPORA if c != heldout},
                       'split_fingerprint': self.split_fingerprint,
                       'manifest_fingerprint': immutable['manifest_fingerprint'],
                       'sizes': {'train': len(self.train_df), 'val': len(self.val_df), 'test': len(self.test_df)},
                       'test_speaker_count': int(self.test_df.speaker_id.nunique()),
                       'python': platform.python_version(), 'pytorch': torch.__version__,
                       'throughput_optimizations': {'eval_microbatch': a.eval_batch_size,
                                                    'data_workers': a.num_workers,
                                                    'gradient_checkpointing': a.gradient_checkpointing,
                                                    'adamw_foreach': True,
                                                    'training_examples_per_epoch': 24*a.updates_per_epoch}}
        old = self.output/'resolved_config.json'
        # Re-running with modified training/preprocessing settings must not silently
        # load an incompatible checkpoint or append to a different experiment.
        if old.exists():
            cfg = json.loads(old.read_text())
            allowed_changes = {}
            for key in ('split_fingerprint', 'checkpoint_revision', 'seed', 'batch', 'gpu_budget',
                        'contrastive_queue_capacity', 'audio', 'learning_stages', 'objective',
                        'early_stopping', 'optimizer_schedule', 'throughput_optimizations'):
                if cfg.get(key) != self.config[key]:
                    if a.resume and a.allow_throughput_resume and key in ('batch', 'gpu_budget', 'throughput_optimizations'):
                        allowed_changes[key] = {'previous': cfg.get(key), 'new': self.config[key]}
                    else:
                        raise RuntimeError(f'{heldout} existing outputs use different {key}; '
                                           'choose NEW --output-dir, or use --resume '
                                           '--allow-throughput-resume for ONLY batch/GPU-budget changes')
            if allowed_changes:
                if not (self.output/'last.pth').is_file():
                    raise RuntimeError('Throughput-resume requested but last.pth is missing')
                if (self.output/'training_complete.json').exists():
                    raise RuntimeError('Cannot change training configuration after completion; '
                                       'run this corpus with its original settings or use a new output directory')
                original = self.output/'original_config_before_fast_resume.json'
                if not original.exists():
                    write_json(original, cfg)
                journal_path = self.output/'throughput_resume_events.json'
                events = json.loads(journal_path.read_text()) if journal_path.exists() else []
                events.append({'time_utc': datetime.now(timezone.utc).isoformat(),
                               'changed': allowed_changes,
                               'caveat': 'Batch sampling and SupCon positive/negative set may change; '
                                         'resume is NOT bitwise identical to the old run. '
                                         'This continues only from last fully completed epoch.'})
                write_json(journal_path, events)
                write_json(old, self.config)
                log.warning('[%s] THROUGHPUT RESUME: changed %s; history and old config archived. '
                            'Sampling/loss trajectory is not identical to the previous run.',
                            heldout, sorted(allowed_changes))
        else:
            write_json(old, self.config)
        for phase, frame in (('train', self.train_df), ('val', self.val_df), ('test', self.test_df)):
            frame[['corpus', 'recording_id', 'speaker_id', 'emotion_id', 'rel_path', 'sha256']].to_csv(
                self.output/f'{phase}_manifest.csv', index=False)
        self.model = None
        self.extractor = None
        self.collate = None
        self.train_ds = None
        self.sampler = None

    def ensure_model(self):
        if self.model is not None:
            return
        if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
            raise RuntimeError('A100/BF16-capable NVIDIA CUDA GPU required to train/test this WavLM-Large model')
        from transformers import AutoFeatureExtractor, WavLMConfig, WavLMModel
        import transformers
        self.log.info('[MODEL] GPU=%s, torch=%s, transformers=%s',
                      torch.cuda.get_device_name(0), torch.__version__, transformers.__version__)
        configure_cuda_budget(self.a.max_gpu_gib, self.a.cuda_overhead_gib, self.log)
        cfg = WavLMConfig.from_pretrained(PRETRAINED_MODEL, revision=self.a.checkpoint_revision)
        if cfg.hidden_size != 1024 or cfg.num_hidden_layers != 24:
            raise RuntimeError('Reference requires 24-layer 1024-hidden WavLM Large')
        cfg.apply_spec_augment = False
        cfg.layerdrop = 0.0
        self.extractor = AutoFeatureExtractor.from_pretrained(PRETRAINED_MODEL,
                                                               revision=self.a.checkpoint_revision)
        if self.extractor.sampling_rate != SAMPLE_RATE or not self.extractor.do_normalize:
            raise RuntimeError('Unexpected WavLM feature extraction specification')
        backbone = WavLMModel.from_pretrained(PRETRAINED_MODEL,
                                              revision=self.a.checkpoint_revision, config=cfg)
        self.model = ArchitectureA(backbone).to(self.device)
        self.model.set_stage('frozen')
        if self.a.gradient_checkpointing:
            try:
                self.model.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
            except TypeError:
                self.model.wavlm.gradient_checkpointing_enable()
        else:
            self.model.wavlm.gradient_checkpointing_disable()
        self.log.info('[THROUGHPUT] WavLM gradient checkpointing: %s (disabled = faster, more activation memory)',
                      'enabled' if self.a.gradient_checkpointing else 'disabled')
        self.collate = collate_factory(self.extractor, self.window)
        initial_mem = gpu_memory_snapshot(self.device)
        self.log.info('[VRAM MODEL LOADED] PyTorch reserved=%.2f GiB; process GPU=%s GiB',
                      initial_mem['reserved_gib'],
                      ('n/a' if initial_mem['process_nvidia_smi_gib'] is None else
                       f"{initial_mem['process_nvidia_smi_gib']:.2f}"))
        if (initial_mem['process_nvidia_smi_gib'] is not None and
                initial_mem['process_nvidia_smi_gib'] > self.a.max_gpu_gib):
            raise RuntimeError('GPU budget exceeded while loading model; use MIG or reduce budget overhead')
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        self.log.info('[MODEL] total params=%d; frozen warmup trainable=%d',
                      sum(p.numel() for p in self.model.parameters()),
                      sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    def optimizer(self, stage):
        groups = defaultdict(list)
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if name.startswith('wavlm.encoder.layers.'):
                idx = int(name.split('.')[3])
                if idx < 18:
                    raise RuntimeError('A lower WavLM encoder layer was unexpectedly unfrozen')
                lr = 5e-6*(.8**(23-idx))
            else:
                lr = 3e-4 if stage == 'frozen' else 1e-4
            decay = .01 if (param.ndim >= 2 and not name.endswith('bias') and
                            'layer_logits' not in name and 'group_gate' not in name) else 0.
            groups[(lr, decay)].append(param)
        return torch.optim.AdamW([
            {'params': params, 'lr': lr, 'weight_decay': wd} for (lr, wd), params in groups.items()],
            betas=(.9, .999), eps=1e-8, foreach=True)

    def scheduler(self, opt, stage):
        epochs = self.a.frozen_epochs if stage == 'frozen' else self.a.max_epochs-self.a.frozen_epochs
        n = self.a.updates_per_epoch*epochs
        warm = max(1, round(n*.10))
        def multiplier(step):
            if step < warm:
                return max(.05, (step+1)/warm)
            progress = min(1., (step-warm)/max(1, n-warm))
            return .05+.95*(1+math.cos(math.pi*progress))/2
        return torch.optim.lr_scheduler.LambdaLR(opt, multiplier)

    def make_train_sampler(self):
        self.train_ds = TrainDataset(self.train_df, self.store, self.window, self.a.seed)
        self.samplers = {}
        for stage, batch_size, grad_accum in (
            ('frozen', self.a.frozen_batch_size, self.a.frozen_grad_accum),
            ('finetune', self.a.batch_size, self.a.grad_accum),
        ):
            sampler = CorpusCrossSpeakerPairSampler(
                self.train_ds, batch_size,
                self.a.updates_per_epoch*grad_accum, self.a.seed)
            # Verify each sampled same-class pair has distinct source corpora.
            for idx, batch in enumerate(sampler):
                by_class = defaultdict(list)
                for i in batch:
                    r = self.train_ds.rows[i]
                    by_class[r['emotion_id']].append(r)
                assert all(len(group) == 2 and group[0]['corpus'] != group[1]['corpus']
                           for group in by_class.values())
                if idx >= 12:
                    break
            self.samplers[stage] = sampler
            self.log.info('[%s] %s stage: microbatch=%d accumulation=%d, effective=24, '
                          'microbatches/epoch=%d, optimizer steps/epoch=%d',
                          self.heldout, stage, batch_size, grad_accum,
                          len(sampler), self.a.updates_per_epoch)
        self.sampler = self.samplers['finetune']

    def infer_recordings(self, df, phase, epoch, collect_embeddings=False):
        self.model.eval()
        ds = EvalDataset(df, self.store, self.window,
                         self.a.eval_hop_fraction, self.a.max_eval_windows)
        infer_batch = self.a.inference_batch_size or self.a.eval_batch_size
        self.log.info('[%s EVAL] num_workers=%d, inference_batch=%d, test_recovery_requested=%s',
                      self.heldout, self.a.eval_workers, infer_batch, self.a.recover_crashed_test)
        loader = DataLoader(ds, batch_size=infer_batch, shuffle=False,
                            num_workers=self.a.eval_workers, pin_memory=True,
                            collate_fn=self.collate,
                            **({'prefetch_factor': 2} if self.a.eval_workers else {}))
        sums, gates, counts = {}, {}, Counter()
        embeddings = {} if collect_embeddings else None
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f'[EVAL {self.heldout}] {phase}',
                              mininterval=2., disable=not sys.stderr.isatty()):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(batch['input_values'].to(self.device, non_blocking=True),
                                         batch['attention_mask'].to(self.device, non_blocking=True))
                logits = outputs['logits'].float().cpu().numpy()
                gates_arr = outputs['group_gates'].float().cpu().numpy()
                vectors = outputs['embedding'].float().cpu().numpy() if collect_embeddings else None
                if not np.isfinite(logits).all() or not np.isfinite(gates_arr).all():
                    raise FloatingPointError('Non-finite inference outputs')
                for i, rid in enumerate(batch['recording_id']):
                    if rid not in sums:
                        sums[rid] = np.zeros(NC, dtype=np.float64)
                        gates[rid] = np.zeros(4, dtype=np.float64)
                        if collect_embeddings:
                            embeddings[rid] = np.zeros(512, dtype=np.float64)
                    sums[rid] += logits[i]
                    gates[rid] += gates_arr[i]
                    if collect_embeddings:
                        embeddings[rid] += vectors[i]
                    counts[rid] += 1
        mem = gpu_memory_snapshot(self.device)
        if mem['process_nvidia_smi_gib'] is not None and mem['process_nvidia_smi_gib'] > self.a.max_gpu_gib:
            raise RuntimeError(f"Evaluation exceeded VRAM target: {mem['process_nvidia_smi_gib']:.2f} GiB")
        if set(counts) != set(df.recording_id):
            raise RuntimeError('Missing/extra recording in evaluation')
        rows = []
        for r in ds.rows:
            rid = r['recording_id']
            logit = sums[rid]/counts[rid]
            prob = softmax(logit)
            row = {'corpus': r['corpus'], 'recording_id': rid, 'speaker_id': r['speaker_id'],
                   'true_class': int(r['emotion_id']), 'pred_class': int(np.argmax(prob)),
                   'n_windows': counts[rid], 'model_epoch': epoch}
            row.update({f'p_{SLUGS[i]}': float(prob[i]) for i in range(NC)})
            row.update({f'logit_{SLUGS[i]}': float(logit[i]) for i in range(NC)})
            row.update({f'ghf_gate_{i+1}': float(gates[rid][i]/counts[rid]) for i in range(4)})
            rows.append(row)
        pred_frame = pd.DataFrame(rows).sort_values('recording_id').reset_index(drop=True)
        if collect_embeddings:
            embedding_mat = np.stack([(embeddings[rid]/counts[rid]).astype('float32')
                                      for rid in pred_frame.recording_id])
            return pred_frame, embedding_mat
        return pred_frame

    def train_epoch(self, epoch, opt, scheduler):
        self.model.train()
        self.train_ds.epoch = epoch
        sampler = self.samplers[self.model.stage]
        sampler.epoch = epoch
        grad_accum = (self.a.frozen_grad_accum if self.model.stage == 'frozen'
                      else self.a.grad_accum)
        stage_batch = (self.a.frozen_batch_size if self.model.stage == 'frozen'
                       else self.a.batch_size)
        loader = DataLoader(self.train_ds, batch_sampler=sampler,
                            num_workers=self.a.num_workers, pin_memory=True,
                            collate_fn=self.collate,
                            **({'prefetch_factor': 2} if self.a.num_workers else {}))
        n_batches = len(loader)
        assert n_batches == self.a.updates_per_epoch*grad_accum
        sup_weight = 0. if epoch <= self.a.frozen_epochs else min(.05, .05*(epoch-self.a.frozen_epochs)/3.)
        queue = SupervisedContrastiveQueue(self.a.contrastive_queue_size) if stage_batch == 2 else None
        # Accumulate detached training diagnostics on GPU. Calling .item() four
        # times per microbatch forces thousands of CUDA/CPU synchronizations.
        totals_gpu = torch.zeros(4, device=self.device, dtype=torch.float32)
        seen = 0
        steps = 0
        opt.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats(self.device)
        start = time.perf_counter()
        bar = tqdm(loader, desc=f'[{self.heldout}] epoch {epoch}/{self.a.max_epochs}',
                   mininterval=2., disable=not sys.stderr.isatty())
        for bi, batch in enumerate(bar):
            input_values = batch['input_values'].to(self.device, non_blocking=True)
            attention_mask = batch['attention_mask'].to(self.device, non_blocking=True)
            y = batch['y'].to(self.device, non_blocking=True)
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                out = self.model(input_values, attention_mask, need_projection=(sup_weight > 0))
            ce = F.cross_entropy(out['logits'].float(), y, label_smoothing=.05)
            con = ((queue.loss(out['projection'], y) if queue is not None
                    else supcon_loss(out['projection'], y))
                   if sup_weight else ce.new_zeros(()))
            loss = ce+sup_weight*con
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch} batch={bi}')
            (loss/grad_accum).backward()
            if queue is not None and sup_weight:
                queue.enqueue(out['projection'], y)
            bs = len(y)
            seen += bs
            with torch.no_grad():
                totals_gpu.add_(torch.stack((
                    loss.detach()*bs,
                    ce.detach()*bs,
                    con.detach()*bs,
                    (out['logits'].detach().argmax(-1) == y).sum().float(),
                )))
            if (bi+1)%grad_accum == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.)
                if not torch.isfinite(grad_norm):
                    raise FloatingPointError('Non-finite gradient norm')
                opt.step()
                scheduler.step()
                opt.zero_grad(set_to_none=True)
                steps += 1
                if steps % 25 == 0 or steps == self.a.updates_per_epoch:
                    mem = gpu_memory_snapshot(self.device)
                    if (mem['process_nvidia_smi_gib'] is not None and
                            mem['process_nvidia_smi_gib'] > self.a.max_gpu_gib):
                        raise RuntimeError('Physical GPU memory exceeded the requested target: '
                                           f"{mem['process_nvidia_smi_gib']:.2f} > {self.a.max_gpu_gib:.2f} GiB. "
                                           'Use a larger CUDA overhead margin or MIG partitioning.')
                    logged = totals_gpu.detach().cpu().tolist()
                    self.log.info('[%s epoch=%d] steps=%d/%d, loss=%.4f, crop_acc=%.3f, '
                                  'GPU reserved=%.2f GiB peak=%.2f GiB, process=%s GiB',
                                  self.heldout, epoch, steps, self.a.updates_per_epoch,
                                  logged[0]/seen, logged[3]/seen,
                                  mem['reserved_gib'], mem['peak_reserved_gib'],
                                  ('n/a' if mem['process_nvidia_smi_gib'] is None else
                                   f"{mem['process_nvidia_smi_gib']:.2f}"))
        assert steps == self.a.updates_per_epoch
        metric_values = totals_gpu.detach().cpu().tolist()
        assert seen == self.a.updates_per_epoch*24
        return {'epoch': epoch, 'stage': self.model.stage,
                'train_loss': metric_values[0]/seen,
                'train_ce': metric_values[1]/seen,
                'train_supcon': metric_values[2]/seen,
                'train_accuracy': metric_values[3]/seen,
                'supcon_weight': sup_weight, 'n_crop_examples': int(seen),
                'updates': steps, 'microbatch': stage_batch, 'grad_accum': grad_accum,
                'seconds': time.perf_counter()-start,
                'peak_gpu_gib': torch.cuda.max_memory_allocated(self.device)/2**30,
                'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(self.device)/2**30}

    def validate(self, epoch):
        preds = self.infer_recordings(self.val_df, 'val', epoch)
        measures = {c: quick_metrics(preds[preds.corpus == c])
                    for c in CORPORA if c != self.heldout}
        out = {'val_macro_corpus_uar': float(np.mean([m['uar'] for m in measures.values()])),
               'val_macro_corpus_f1': float(np.mean([m['macro_f1'] for m in measures.values()])),
               'val_macro_corpus_acc': float(np.mean([m['accuracy'] for m in measures.values()])),
               'val_loss': float(np.mean([m['cross_entropy'] for m in measures.values()]))}
        for corpus, m in measures.items():
            out[f'val_uar_{corpus}'] = m['uar']
            out[f'val_f1_{corpus}'] = m['macro_f1']
            out[f'val_acc_{corpus}'] = m['accuracy']
        return out, preds

    def cp_common(self, epoch, metrics):
        # Convert entire state dict to FP32 on CPU for portable exact-fidelity loads.
        weights = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
        return {'format': FORMAT, 'protocol': PROTOCOL, 'heldout': self.heldout,
                'split_fingerprint': self.split_fingerprint,
                'checkpoint_revision': self.a.checkpoint_revision,
                'epoch': int(epoch), 'stage': self.model.stage,
                'model': weights, 'validation': metrics}

    def verify_checkpoint(self, checkpoint):
        if checkpoint.get('format') != FORMAT or checkpoint.get('protocol') != PROTOCOL:
            raise RuntimeError('Checkpoint format/protocol mismatch')
        if checkpoint['heldout'] != self.heldout or checkpoint['split_fingerprint'] != self.split_fingerprint:
            raise RuntimeError('Checkpoint belongs to another test corpus or source split')
        if checkpoint['checkpoint_revision'] != self.a.checkpoint_revision:
            raise RuntimeError('Pretrained WavLM revision changed')

    def load_model_checkpoint(self, path):
        self.ensure_model()
        cp = torch.load(path, map_location='cpu', weights_only=False)
        self.verify_checkpoint(cp)
        self.model.load_state_dict(cp['model'], strict=True)
        self.model.set_stage(cp['stage'])
        return cp

    def save_last(self, epoch, opt, sched, validation, history, best_rank, best_epoch, patience):
        cp = self.cp_common(epoch, validation)
        cp.update({'optimizer': opt.state_dict(), 'scheduler': sched.state_dict(),
                   'history': history, 'best_rank': best_rank, 'best_epoch': best_epoch,
                   'patience_count': patience,
                   'rng_python': random.getstate(), 'rng_numpy': np.random.get_state(),
                   'rng_torch': torch.get_rng_state(), 'rng_cuda': torch.cuda.get_rng_state_all()})
        atomic_torch_save(self.output/'last.pth', cp)
        del cp

    def train(self):
        done = self.output/'training_complete.json'
        if done.exists():
            self.log.info('[%s] Training already complete; not retraining', self.heldout)
            return
        last = self.output/'last.pth'
        if last.exists() and not self.a.resume:
            raise RuntimeError(f'Found interrupted training in {last}. Add --resume or choose fresh --output-dir')
        self.ensure_model()
        self.make_train_sampler()
        history, best_rank, best_epoch, patience = [], None, 0, 0
        patience_ref = -1.
        start_epoch = 1
        last_stage, resume = None, None
        if last.exists():
            resume = self.load_model_checkpoint(last)
            history = resume['history']
            best_rank = tuple(resume['best_rank']) if resume['best_rank'] is not None else None
            best_epoch = int(resume['best_epoch'])
            patience = int(resume['patience_count'])
            patience_ref = max((x['val_macro_corpus_uar'] for x in history), default=-1.)
            start_epoch = int(resume['epoch'])+1
            random.setstate(resume['rng_python'])
            np.random.set_state(resume['rng_numpy'])
            torch.set_rng_state(resume['rng_torch'])
            torch.cuda.set_rng_state_all(resume['rng_cuda'])
            if best_epoch:
                best_cp = torch.load(self.output/'best.pth', map_location='cpu', weights_only=False)
                self.verify_checkpoint(best_cp)
                if int(best_cp['epoch']) != best_epoch:
                    backup = self.output/'best_previous.pth'
                    if not backup.exists():
                        raise RuntimeError('Uncommitted best checkpoint and no backup; safe resume impossible')
                    previous = torch.load(backup, map_location='cpu', weights_only=False)
                    self.verify_checkpoint(previous)
                    if int(previous['epoch']) != best_epoch:
                        raise RuntimeError('Best/last checkpoint inconsistency; safe resume impossible')
                    shutil.copy2(backup, self.output/'best.pth')
                    self.log.warning('[%s] Rolled back uncommitted best checkpoint to epoch %d',
                                     self.heldout, best_epoch)
                    del previous
                del best_cp
            self.log.info('[%s] Resuming at epoch %d; best epoch=%d',
                          self.heldout, start_epoch, best_epoch)
        if best_epoch and not (self.output/'best.pth').exists():
            raise RuntimeError('Best checkpoint missing; cannot safely resume')
        opt, sched = None, None
        for epoch in range(start_epoch, self.a.max_epochs+1):
            stage = 'frozen' if epoch <= self.a.frozen_epochs else 'finetune'
            if stage != last_stage:
                self.model.set_stage(stage)
                opt = self.optimizer(stage)
                sched = self.scheduler(opt, stage)
                if resume is not None and resume['stage'] == stage:
                    opt.load_state_dict(resume['optimizer'])
                    sched.load_state_dict(resume['scheduler'])
                    # Older low-memory checkpoints may store foreach=False;
                    # the fast variant intentionally re-enables batched AdamW.
                    for param_group in opt.param_groups:
                        param_group['foreach'] = True
                last_stage = stage
                resume = None
            train_row = self.train_epoch(epoch, opt, sched)
            val_row, val_pred = self.validate(epoch)
            row = {**train_row, **val_row}
            history.append(row)
            pd.DataFrame(history).to_csv(self.output/'train_history.csv', index=False)
            plot_training(history, self.output, self.heldout)
            # Selection follows SOURCE VALIDATION ONLY. Test audio NEVER enters here.
            rank = (round(val_row['val_macro_corpus_uar'], 12),
                    round(val_row['val_macro_corpus_f1'], 12),
                    round(val_row['val_macro_corpus_acc'], 12),
                    -val_row['val_loss'], -epoch)
            if best_rank is None or rank > best_rank:
                best_file = self.output/'best.pth'
                if best_file.exists():
                    shutil.copy2(best_file, self.output/'best_previous.pth')
                best_rank, best_epoch = rank, epoch
                cp = self.cp_common(epoch, val_row)
                atomic_torch_save(best_file, cp)
                del cp
                val_pred.to_csv(self.output/'best_validation_predictions.csv', index=False)
                self.log.info('[%s] New best epoch=%d SOURCE VAL mean-UAR=%.4f',
                              self.heldout, epoch, val_row['val_macro_corpus_uar'])
            if val_row['val_macro_corpus_uar'] > patience_ref+self.a.min_delta:
                patience_ref = val_row['val_macro_corpus_uar']
                patience = 0
            elif epoch > self.a.frozen_epochs:
                patience += 1
            self.save_last(epoch, opt, sched, val_row, history, best_rank, best_epoch, patience)
            self.log.info('[%s epoch %d] val_UAR=%.2f%% val_F1=%.2f%% best_epoch=%d stop=%d/%d GPU=%.2fGiB',
                          self.heldout, epoch, 100*val_row['val_macro_corpus_uar'],
                          100*val_row['val_macro_corpus_f1'], best_epoch, patience,
                          self.a.patience, row['peak_gpu_gib'])
            if epoch > self.a.frozen_epochs and patience >= self.a.patience:
                self.log.info('[%s] Early stop after epoch %d', self.heldout, epoch)
                break
        if best_epoch == 0:
            raise RuntimeError('Training completed without a best checkpoint')
        best = self.load_model_checkpoint(self.output/'best.pth')
        val_pred, val_emb = self.infer_recordings(self.val_df, 'val', best_epoch, collect_embeddings=True)
        (self.output/'validation').mkdir(parents=True, exist_ok=True)
        np.save(self.output/'validation'/'source_validation_embeddings_512d.npy', val_emb, allow_pickle=False)
        val_pred[['recording_id', 'speaker_id', 'true_class', 'corpus']].to_csv(
            self.output/'validation'/'source_validation_embedding_ids.csv', index=False)
        # Report per-source validation figures without exposing the target corpus.
        for corpus in [c for c in CORPORA if c != self.heldout]:
            p = val_pred[val_pred.corpus == corpus].reset_index(drop=True)
            export_scores(p, self.output/'validation'/corpus, f'{corpus.upper()} source validation',
                          self.a.bootstrap_repeats, self.a.seed)
        export_scores(val_pred, self.output/'validation'/'pooled_sources',
                      'Pooled source validation (NOT target test)',
                      self.a.bootstrap_repeats, self.a.seed)
        weights = F.softmax(self.model.ghf.layer_logits.float(), dim=-1).detach().cpu().numpy()
        layer_table = pd.DataFrame([{'layer': gi*6+j+1, 'group': gi+1, 'weight': float(weights[gi, j])}
                                    for gi in range(4) for j in range(6)])
        layer_table.to_csv(self.output/'ghf_layer_weights.csv', index=False)
        plot_heatmap(weights, [f'Layer {j+1}' for j in range(6)],
                     [f'Group {g+1}' for g in range(4)], self.output/'ghf_layer_weights.png',
                     f'{self.heldout}: relative weights within 6-layer GHF groups', percentages=True)
        write_json(done, {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                          'experiment': self.heldout, 'best_epoch': best_epoch,
                          'best_source_validation': best['validation'],
                          'best_checkpoint_sha256': digest_file(self.output/'best.pth'),
                          'split_fingerprint': self.split_fingerprint,
                          'note': 'Target test corpus was not evaluated during model selection'})
        self.log.info('[%s] Training finished; best checkpoint epoch=%d', self.heldout, best_epoch)

    def test(self):
        if not self.a.unlock_test:
            raise PermissionError('Explicit --unlock-test required')
        done = self.output/'training_complete.json'
        if not done.exists():
            raise RuntimeError(f'{self.heldout}: training must finish before its held-out test')
        receipt = self.output/'final_test_receipt.json'
        if receipt.exists():
            self.log.info('[%s] Locked test already done; not repeating', self.heldout)
            return
        test_in_progress = self.output/'final_test_started.lock'
        meta = json.loads(done.read_text())
        best_path = self.output/'best.pth'
        sha = digest_file(best_path)
        if sha != meta['best_checkpoint_sha256'] or meta['split_fingerprint'] != self.split_fingerprint:
            raise RuntimeError('Best checkpoint or split changed after training')
        if test_in_progress.exists():
            if not self.a.recover_crashed_test:
                raise RuntimeError('A final test was started but not completed. '
                                   'After confirming the crash yielded no saved predictions, '
                                   'use --recover-crashed-test to archive the attempt and retry.')
            recover_crashed_final_test(self.output, test_in_progress, self.log, self.heldout, sha)
        cp = self.load_model_checkpoint(best_path)
        with open(test_in_progress, 'x', encoding='utf8') as lock_fd:
            lock_fd.write(datetime.now(timezone.utc).isoformat()+'\n')
        preds, test_embeddings = self.infer_recordings(
            self.test_df, 'test_final_once', int(cp['epoch']), collect_embeddings=True)
        if len(preds) != len(self.test_df):
            raise RuntimeError('Test predictions do not cover ALL held-out original recordings')
        metrics = export_scores(preds, self.output/'test',
                                f'LOCO: unseen {self.heldout.upper()} test corpus',
                                self.a.bootstrap_repeats, self.a.seed)
        np.save(self.output/'test'/'recording_embeddings_512d.npy', test_embeddings, allow_pickle=False)
        preds[['recording_id', 'speaker_id', 'true_class', 'corpus']].to_csv(
            self.output/'test'/'embedding_recording_ids.csv', index=False)
        plot_embedding_pca(test_embeddings, preds, self.output/'test'/'embedding_pca_2d.png',
                           f'{self.heldout.upper()} unseen corpus: 512-D embedding PCA')
        metrics.update({'held_out_corpus': self.heldout, 'best_epoch': int(cp['epoch']),
                        'source_val_mean_corpus_uar': cp['validation']['val_macro_corpus_uar'],
                        'checkpoint_sha256': sha, 'split_fingerprint': self.split_fingerprint,
                        'test_protocol': 'whole unseen corpus, all available speakers, once'})
        write_json(self.output/'test'/'metrics.json', metrics)
        write_json(receipt, {'executed_at_utc': datetime.now(timezone.utc).isoformat(),
                             'held_out_corpus': self.heldout, 'checkpoint_sha256': sha,
                             'test_evaluated_once': True, 'test_recordings': len(preds),
                             'accuracy': metrics['accuracy'], 'uar': metrics['uar'],
                             'macro_f1': metrics['macro_f1']})
        self.log.info('[FINAL TEST %s] accuracy=%.2f%% UAR=%.2f%% macro-F1=%.2f%% n=%d',
                      self.heldout, metrics['accuracy']*100, metrics['uar']*100,
                      metrics['macro_f1']*100, len(preds))

    def close(self):
        if self.model is not None:
            del self.model
            self.model = None
        self.extractor = None
        self.train_ds = None
        self.sampler = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def write_comparison_report(a, log):
    report_dir = a.output_dir/'paper_results'
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, class_rows, confusions, gate_rows = [], [], {}, []
    for heldout in CORPORA:
        path = a.output_dir/('test_'+heldout.replace('-', '_'))/'test'
        met = path/'metrics.json'
        if not met.exists():
            log.warning('[REPORT] %s has no finished final test; omitting it', heldout)
            continue
        metrics = json.loads(met.read_text())
        rows.append({'Test corpus': heldout.upper(), 'Train corpora': ' + '.join(c.upper() for c in CORPORA if c != heldout),
                     'N': metrics['n_recordings'], 'Speakers': metrics['n_speakers'],
                     'Accuracy': metrics['accuracy'], 'UAR': metrics['uar'],
                     'Macro F1': metrics['macro_f1'], 'Weighted F1': metrics['weighted_f1'],
                     'AUROC OvR': metrics['auroc_macro_ovr'],
                     'Cross entropy': metrics['cross_entropy'], 'ECE': metrics['ece_15_bins'],
                     'Accuracy CI low': metrics['accuracy_95ci_speaker_bootstrap'][0],
                     'Accuracy CI high': metrics['accuracy_95ci_speaker_bootstrap'][1],
                     'UAR CI low': metrics['uar_95ci_speaker_bootstrap'][0],
                     'UAR CI high': metrics['uar_95ci_speaker_bootstrap'][1],
                     'Best epoch': metrics['best_epoch'],
                     'Source validation mean UAR': metrics['source_val_mean_corpus_uar']})
        class_df = pd.read_csv(path/'per_class_metrics.csv')
        for _, r in class_df.iterrows():
            class_rows.append({'test_corpus': heldout, 'emotion': r.emotion,
                               'support': int(r.support), 'precision': r.precision,
                               'recall': r.recall, 'f1': r.f1})
        confusions[heldout] = pd.read_csv(path/'confusion_normalized.csv', index_col=0).to_numpy()
        preds = pd.read_csv(path/'recording_predictions.csv')
        for i in range(4):
            gate_rows.append({'test_corpus': heldout, 'group': i+1,
                              'mean_gate': float(preds[f'ghf_gate_{i+1}'].mean())})
    if not rows:
        log.warning('[REPORT] No completed tests; first run --mode all --unlock-test')
        return
    df = pd.DataFrame(rows)
    class_df = pd.DataFrame(class_rows)
    df.to_csv(report_dir/'table_1_cross_corpus_metrics.csv', index=False)
    class_df.to_csv(report_dir/'table_2_per_class_metrics.csv', index=False)
    pd.DataFrame(gate_rows).to_csv(report_dir/'table_3_mean_group_gates.csv', index=False)
    pd.DataFrame({metric: df[metric].describe() for metric in ('Accuracy', 'UAR', 'Macro F1')}).to_csv(
        report_dir/'table_4_across_corpus_descriptive_statistics.csv')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    xs = np.arange(len(df))
    for j, (metric, display) in enumerate([('Accuracy', 'Accuracy'), ('UAR', 'UAR'), ('Macro F1', 'Macro-F1')]):
        ax.bar(xs+(j-1)*.23, df[metric]*100, width=.23, label=display)
    ax.set_xticks(xs, df['Test corpus'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Percent')
    ax.set_title('Architecture A: four leave-one-corpus-out experiments')
    ax.axhline(100/NC, linestyle='--', color='gray', label='Uniform random chance (16.7%)')
    ax.legend()
    ax.grid(axis='y', alpha=.15)
    fig.tight_layout()
    fig.savefig(report_dir/'figure_1_cross_corpus_metrics.png', dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5))
    errors = np.vstack((100*(df['Accuracy']-df['Accuracy CI low']),
                        100*(df['Accuracy CI high']-df['Accuracy'])))
    ax.bar(xs, 100*df.Accuracy, yerr=errors, capsize=5)
    ax.set_xticks(xs, df['Test corpus'])
    ax.set_ylim(0, 105)
    ax.set_ylabel('Accuracy (%) and 95% speaker-bootstrap CI')
    ax.set_title('Cross-corpus accuracy uncertainty (speaker-level resampling)')
    fig.tight_layout()
    fig.savefig(report_dir/'figure_2_accuracy_with_speaker_bootstrap_ci.png', dpi=220)
    plt.close(fig)
    for field, fig_num in [('recall', 3), ('f1', 4)]:
        piv = class_df.pivot(index='test_corpus', columns='emotion', values=field).reindex(
            index=[c for c in CORPORA if c in class_df.test_corpus.unique()], columns=list(LABELS))
        plot_heatmap(piv.to_numpy(), LABELS, [s.upper() for s in piv.index],
                     report_dir/f'figure_{fig_num}_per_class_{field}_by_test_corpus.png',
                     f'Per-emotion {field} in each unseen test corpus', percentages=True)
    if len(confusions) == 4:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), constrained_layout=True)
        for ax, corpus in zip(axes.flat, CORPORA):
            mat = confusions[corpus]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap='Blues')
            for i in range(NC):
                for j in range(NC):
                    ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=7,
                            color='white' if mat[i,j] > .65 else 'black')
            ax.set_xticks(range(NC), LABELS, rotation=45, ha='right', fontsize=8)
            ax.set_yticks(range(NC), LABELS, fontsize=8)
            ax.set_title(corpus.upper())
            ax.set_xlabel('Predicted'); ax.set_ylabel('Actual')
        fig.colorbar(im, ax=axes.ravel().tolist(), shrink=.7)
        fig.savefig(report_dir/'figure_5_four_normalized_confusions.png', dpi=220)
        plt.close(fig)
    gate = pd.DataFrame(gate_rows)
    pivot = gate.pivot(index='test_corpus', columns='group', values='mean_gate').reindex(
        index=[c for c in CORPORA if c in gate.test_corpus.unique()])
    plot_heatmap(pivot.to_numpy(), [f'Group {i}' for i in range(1, 5)],
                 [s.upper() for s in pivot.index], report_dir/'figure_6_ghf_group_gates.png',
                 'Average data-dependent GHF gates on unseen corpus', percentages=True)
    # Paper-ready Markdown and LaTeX, with no fabricated values.
    md = ['# Architecture A: strict cross-corpus six-class speech emotion recognition', '',
          'Training source corpora: three; validation: speaker-disjoint recordings from each source;',
          'testing: the entire fourth corpus, only once after validation checkpoint selection.', '',
          '**Shared labels:** '+', '.join(LABELS), '',
          '## Test results', '', '| Held-out corpus | N | Accuracy | UAR | Macro F1 | Accuracy 95% CI |',
          '|---|---:|---:|---:|---:|---:|']
    for _, r in df.iterrows():
        md.append(f"| {r['Test corpus']} | {int(r['N'])} | {100*r['Accuracy']:.2f}% | "
                  f"{100*r['UAR']:.2f}% | {100*r['Macro F1']:.2f}% | "
                  f"[{100*r['Accuracy CI low']:.2f}%, {100*r['Accuracy CI high']:.2f}%] |")
    md += ['', '## Reporting notes',
           '- All percentages in this table are computed from original recordings, not audio windows.',
           '- UAR is macro-average recall across six shared classes; macro-F1 likewise weights classes equally.',
           '- CIs are speaker-cluster bootstraps, not seed-to-seed variability. For EMOVO (six speakers), expect wide CIs.',
           '- Results across corpora reflect language, speakers, acoustics and label-subjectivity shifts simultaneously.',
           '- This is **one fixed seed**, not a multi-seed estimate. Repeat with independent pre-registered seeds for variance.',
           '- Validation speakers belong only to training-source corpora. No target-corpus adaptation, tuning or leakage.',
           '- Experiments are not directly comparable to PME4 four-class, speaker-overlapping trials.',
           '- Dataset-level labels may differ semantically despite harmonized emotion names.',
           '- Excluded: RAVDESS calm/surprise, EMOVO surprise, EmoDB boredom.',
           '- WavLM initialized with pretrained weights; the GHF, DPTP and classifier are trained on source labels.',
           '', '## Figures',
           '1. figure_1_cross_corpus_metrics.png — accuracy, UAR and macro-F1 across the four target corpora.',
           '2. figure_2_accuracy_with_speaker_bootstrap_ci.png — uncertainty with speaker-cluster resampling.',
           '3. figure_3_per_class_recall_by_test_corpus.png — recall heatmap by class and corpus.',
           '4. figure_4_per_class_f1_by_test_corpus.png — macro-class F1 heatmap.',
           '5. figure_5_four_normalized_confusions.png — corpus-level normalized confusion matrices.',
           '6. figure_6_ghf_group_gates.png — average GHF gating responses.',
           '7. Each experiment contains learning_curves.png, source-validation plots, ROC/PR and reliability figures.',
           '']
    (report_dir/'RESEARCH_REPORT.md').write_text('\n'.join(md), encoding='utf8')
    latex = df[['Test corpus', 'N', 'Accuracy', 'UAR', 'Macro F1']].copy()
    for name in ('Accuracy', 'UAR', 'Macro F1'):
        latex[name] = latex[name].map(lambda x: f'{100*x:.2f}\\%')
    (report_dir/'table_1_cross_corpus_metrics.tex').write_text(
        latex.to_latex(index=False, escape=False, caption='Leave-one-corpus-out test results (six shared classes).',
                       label='tab:arch_a_loco'), encoding='utf8')
    write_json(report_dir/'report_manifest.json', {'finished_test_corpora': list(df['Test corpus']),
        'n_complete': len(df), 'required_complete': 4, 'all_complete': len(df) == 4,
        'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'scores_are_real_experiment_outputs': True})
    log.info('[REPORT] %d / 4 experiments available. Tables, figures, LaTeX and Markdown: %s',
             len(df), report_dir)


def run_self_test():
    """CPU-only checks: parsers, 6-class labels, windows, sampler, metrics, GHF/DPTP.

    Does NOT test the pretrained WavLM download, full A100 training, or real data.
    """
    assert parse_record('ravdess', Path('03-01-04-02-01-01-01.wav')) == ('01', 'sad')
    assert parse_record('ravdess', Path('03-01-02-01-01-01-01.wav')) is None
    assert parse_record('crema-d', Path('1001_DFA_FEA_XX.wav')) == ('1001', 'fear')
    assert parse_record('emodb', Path('03a01Fa.wav')) == ('03', 'happy')
    assert parse_record('emodb', Path('03a01Ea.wav')) == ('03', 'disgust')
    assert parse_record('emodb', Path('03a01La.wav')) is None
    assert parse_record('emovo', Path('/a/emovo/EMOVO/f1/gio-f1-b1.wav')) == ('f1', 'happy')
    assert parse_record('emovo', Path('/a/emovo/EMOVO/m3/pau-m3-d1.wav')) == ('m3', 'fear')
    # Must detect ALL failures without aborting at the first duplicate.
    base = {'corpus':'ravdess', 'speaker_id':'ravdess:07', 'emotion':'happy',
            'recording_id':'ravdess:rep1', 'abs_path':'/fake/repetition1.wav',
            'sha256':'SHA_A', 'pcm_sha256':'PCM_A'}
    repeated = {**base, 'recording_id':'ravdess:rep2',
                'abs_path':'/fake/repetition2.wav'}
    conflicting_label = {**base, 'corpus':'crema-d', 'speaker_id':'crema-d:1006',
            'emotion':'neutral', 'recording_id':'crema-d:1006_TIE_NEU_XX',
            'abs_path':'/fake/1006_TIE_NEU_XX.wav', 'sha256':'SHA_B', 'pcm_sha256':'PCM_B'}
    other_label = {**conflicting_label, 'emotion':'happy',
            'recording_id':'crema-d:1006_TIE_HAP_XX',
            'abs_path':'/fake/1006_TIE_HAP_XX.wav'}
    cross_corpus = {**base, 'corpus':'emovo', 'speaker_id':'emovo:f1',
            'recording_id':'emovo:f1-copy', 'abs_path':'/fake/emovo-copy.wav',
            'sha256':'SHA_C', 'pcm_sha256':'PCM_A'}
    different_speaker = {**base, 'speaker_id':'ravdess:08',
            'recording_id':'ravdess:other-actor', 'abs_path':'/fake/other_actor.wav',
            'sha256':'SHA_D', 'pcm_sha256':'PCM_D'}
    different_speaker2 = {**different_speaker, 'speaker_id':'ravdess:09',
            'recording_id':'ravdess:another-actor', 'abs_path':'/fake/another_actor.wav'}
    same_id_other_audio = {**base, 'recording_id':'emodb:03a01Na',
            'corpus':'emodb', 'speaker_id':'emodb:03', 'emotion':'neutral',
            'abs_path':'/fake/id_copy1.wav', 'sha256':'SHA_E', 'pcm_sha256':'PCM_E'}
    same_id_other_audio2 = {**same_id_other_audio, 'abs_path':'/fake/id_copy2.wav',
            'sha256':'SHA_F', 'pcm_sha256':'PCM_F'}
    kept, removed, groups, details, actions = resolve_duplicate_components([
         base, repeated, conflicting_label, other_label, cross_corpus,
         different_speaker, different_speaker2, same_id_other_audio, same_id_other_audio2])
    assert len(kept) == 0  # 'base' also removed due to cross-corpus copy
    assert len(removed) == 9
    reasons = {r['reason'] for r in groups}
    assert any('cross_corpus_audio_overlap' in x for x in reasons)
    assert any('cross_speaker_audio_overlap' in x for x in reasons)
    assert any('conflicting_emotion_labels' in x for x in reasons)
    assert any('same_recording_id_different_audio' in x for x in reasons)
    assert all(r['action']=='quarantined_conflict' for r in removed)
    clean = [{**base, 'sha256':'G1', 'pcm_sha256':'P1'},
             {**repeated, 'sha256':'G1', 'pcm_sha256':'P1'}]
    k, d, g, _, _ = resolve_duplicate_components(clean)
    assert len(k)==1 and len(d)==1 and d[0]['action']=='redundant_duplicate'
    # Same recording ID and different WAVs -> quarantine BOTH, not just the second.
    k, d, _, _, _ = resolve_duplicate_components([same_id_other_audio,same_id_other_audio2])
    assert len(k)==0 and len(d)==2
    assert len(eval_slices(16000, 48000)) == 1
    assert eval_slices(48000, 48000) == [(0, 48000)]
    assert eval_slices(64000, 48000)[-1][-1] == 64000
    assert len(eval_slices(300000, 48000, .5, 5)) == 5
    with torch.no_grad():
        fusion = GroupedHierarchicalFusion(64)
        states = [torch.randn(2, 10, 64) for _ in range(24)]
        valid = torch.tensor([[1]*10, [1]*7+[0]*3], dtype=torch.bool)
        out, weights, gates = fusion(states, valid)
        assert out.shape == (2, 10, 64) and gates.shape == (2, 4)
        assert torch.allclose(weights.sum(-1), torch.ones(4))
        pooled = DualPathTemporalPooling(64, heads=8)(out, valid)
        assert pooled.shape == (2, 512)
        z = torch.randn(6, 128)
        y = torch.tensor([0, 0, 1, 1, 2, 2])
        assert torch.isfinite(supcon_loss(z, y))
        with torch.enable_grad():
            q = SupervisedContrastiveQueue(capacity=12)
            x = torch.randn(2,128,requires_grad=True)
            a = torch.tensor([0,0])
            assert q.loss(x,a).item() == 0.
            q.enqueue(x,a)
            q.enqueue(torch.randn(2,128),torch.tensor([1,1]))
            q_loss = q.loss(x,a)
            assert torch.isfinite(q_loss) and q_loss.item()>0.
            q_loss.backward()
            assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0
    records = []
    for c in ('ravdess', 'crema-d', 'emodb'):
        for s in range(3):
            for emo in range(NC):
                for r in range(2):
                    records.append({'corpus': c, 'speaker_id': f'{c}:{s}',
                                    'emotion_id': emo, 'recording_id': f'{c}-{s}-{emo}-{r}'})
    dummy = type('DummyDataset', (), {'rows': records})()
    for micro, expected_emotions in ((2,1),(6,3),(12,6)):
        sampler = CorpusCrossSpeakerPairSampler(dummy, micro, 12, 123)
        counts = Counter()
        for batch in sampler:
            pairs = defaultdict(list)
            for ix in batch:
                row = records[ix]
                pairs[row['emotion_id']].append(row)
                counts[row['emotion_id']] += 1
            assert len(pairs) == expected_emotions
            assert all(len(items) == 2 and items[0]['corpus'] != items[1]['corpus']
                       for items in pairs.values())
        assert len(set(counts.values())) == 1, counts
    fake = pd.DataFrame({'corpus': ['emovo']*12, 'speaker_id': ['emovo:f1']*6+['emovo:m1']*6,
                         'true_class': list(range(6))*2, 'pred_class': list(range(6))*2,
                         **{f'p_{slug}': [0.999 if j == i else .0002 for i in list(range(6))*2]
                            for j, slug in enumerate(SLUGS)}})
    # Fix test probabilities to sum approximately to one.
    mat = fake[[f'p_{s}' for s in SLUGS]].to_numpy()
    mat = mat/mat.sum(1, keepdims=True)
    for i, s in enumerate(SLUGS):
        fake[f'p_{s}'] = mat[:, i]
    metrics, speakers, classes, cm = classification_metrics(fake, 100, 42)
    assert metrics['accuracy'] == 1. and metrics['uar'] == 1. and cm.trace() == len(fake)
    print('[SELF-TEST PASSED] six class parsers, windowing, stage-aware 12x2 / 6x4 sampler, '
          'CPU contrastive memory bank with gradient, full-scan duplicate quarantine, GHF/DPTP, metrics')


def main(a):
    verify_numpy_torch_compatibility()
    if a.mode == 'self-test':
        return run_self_test()
    a.output_dir = a.output_dir.expanduser().resolve()
    a.data_root = a.data_root.expanduser().resolve()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    log = make_logger(a.output_dir)
    log.info('[RUN] protocol=%s, mode=%s, target=%s, data=%s',
             PROTOCOL, a.mode, a.experiment, a.data_root)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(a.seed)
    if a.mode == 'report':
        return write_comparison_report(a, log)
    if a.skip_wav_hash and a.mode != 'audit':
        raise RuntimeError('Research training/testing requires content hashes; remove --skip-wav-hash.')
    df, val_speakers, immutable = audit_corpora(a, log)
    if a.mode == 'audit':
        return
    targets = CORPORA if a.experiment == 'all' else (a.experiment,)
    for heldout in targets:
        # Same deterministic initialization per experiment, independent of loop order.
        random.seed(a.seed)
        np.random.seed(a.seed)
        torch.manual_seed(a.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(a.seed)
        log.info('='*78)
        log.info('EXPERIMENT HELD OUT: %s; SOURCE TRAINING: %s', heldout,
                 ', '.join(c for c in CORPORA if c != heldout))
        ex = Experiment(a, df, val_speakers, immutable, heldout, log)
        try:
            if a.mode in ('train', 'all'):
                ex.train()
            if a.mode in ('test', 'all'):
                ex.test()
        finally:
            ex.close()
    write_comparison_report(a, log)
    log.info('[DONE] See %s for experiment folders, audit, metrics, figures, and paper tables.', a.output_dir)


if __name__ == '__main__':
    arguments = args_parser()
    try:
        main(arguments)
    except KeyboardInterrupt:
        print('[INTERRUPTED] Resume with --resume. Finished test receipts remain locked.', file=sys.stderr)
        sys.exit(130)
    except torch.cuda.OutOfMemoryError:
        traceback.print_exc()
        print('[CUDA OOM] The fast preset exceeded the chosen VRAM allocator cap. '
              'Use --batch-size 2 --grad-accum 12 and --resume --allow-throughput-resume '
              'to continue from last completed epoch, or lower --eval-batch-size to 6. '
              'VRAM budgets are not hardware quotas.',
              file=sys.stderr)
        sys.exit(2)
    except Exception:
        traceback.print_exc()
        try:
            output = arguments.output_dir.expanduser().resolve()
            output.mkdir(parents=True, exist_ok=True)
            with open(output/'run_failure.txt', 'a') as f:
                f.write(f'\n[{datetime.now(timezone.utc).isoformat()}]\n')
                f.write(traceback.format_exc())
        except Exception:
            pass
        sys.exit(1)
