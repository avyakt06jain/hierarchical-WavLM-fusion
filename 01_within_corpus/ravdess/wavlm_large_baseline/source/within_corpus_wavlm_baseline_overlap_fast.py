#!/usr/bin/env python3
"""Within-corpus speech emotion recognition with per-speaker 70/15/15 recording splits.

Four independent experiments (RAVDESS, CREMA-D, EmoDB, EMOVO). EVERY speaker
occurs in training, validation, and testing. Recordings are strictly disjoint;
fractions are rounded to whole recordings for each speaker.

Common six emotion classes: Neutral, Sad, Angry, Happy, Fear, Disgust.
Same label maps, duplicate quarantine, 16k preprocessing and 3s windows as LOCO.

Modes:
  --mode self-test  (CPU checks)
  --mode audit      (manifest and split audit)
  --mode train      (all four, or choose --experiment)
  --mode test --unlock-test  (one-time final original-recording evaluation)
  --mode report     (tables and plots for completed final tests)

Uses NVIDIA BF16 GPU for train/test and NumPy 1.x compatible with installed PyTorch.
The two standalone scripts share the exact deterministic split algorithm.
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
PROTOCOL = 'WITHIN_CORPUS_EACH_SPEAKER_70_15_15_SIX_CLASS_V2'
AUDIT_POLICY_VERSION = 'FULL_SCAN_QUARANTINE_AMBIGUOUS_GROUPS_V2'
FORMAT = 'WITHIN_WAVLM_MEANPOOL_BASELINE_6CLASS_CHECKPOINT_V1'

ARCH_NAME = 'baseline'

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
                   default=Path('/workspace/avyakt-sc-final/sc-final-speech-model/outputs/within_corpus_wavlm_baseline_overlap_fast'))
    p.add_argument('--experiment', choices=['all', *CORPORA], default='all',
                   help='Select dataset for its own train/val/test experiment or run all four')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window-seconds', type=float, default=3.0)
    p.add_argument('--eval-hop-fraction', type=float, default=0.5)
    p.add_argument('--max-eval-windows', type=int, default=5)
    p.add_argument('--batch-size', type=int, choices=[2, 6, 12], default=12,
                   help='Default 12 for throughput; fall back to 6 or 2 if VRAM requires.')
    p.add_argument('--grad-accum', type=int, default=None)
    p.add_argument('--eval-batch-size', type=int, default=12)
    p.add_argument('--max-gpu-gib', type=float, default=13.5,
                   help='Process VRAM target, 13.5 GiB ~14.50 decimal GB; advisory outside PyTorch allocator.')
    p.add_argument('--cuda-overhead-gib', type=float, default=1.5,
                   help='Headroom for CUDA context/non-PyTorch allocations (default 1.5 GiB).')
    p.add_argument('--updates-per-epoch', type=int, default=0,
                   help='0=automatic per corpus ceil(train_recordings/24); positive overrides.')
    p.add_argument('--max-epochs', type=int, default=28)
    p.add_argument('--frozen-epochs', type=int, default=4)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--min-delta', type=float, default=0.002)
    p.add_argument('--num-workers', type=int, default=4,
                   help='Parallel CPU audio loading; set 0 if multiprocessing is unavailable.')
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
    p.add_argument('--unlock-test', action='store_true',
                   help='Required for --mode test/all; one-time test of held-out recordings per dataset.')
    p.add_argument('--checkpoint-revision', default=DEFAULT_REVISION)
    p.add_argument('--max-file-seconds', type=float, default=30.0)
    p.add_argument('--min-file-seconds', type=float, default=0.3)
    p.add_argument('--bootstrap-repeats', type=int, default=5000)
    a = p.parse_args()
    if a.grad_accum is None:
        a.grad_accum = {2: 12, 6: 4, 12: 2}[a.batch_size]
    if a.batch_size * a.grad_accum != 24:
        p.error('Effective batch must be 24: 2x12, 6x4, or 12x2')
    if not (3.0 <= a.max_gpu_gib <= 80.0) or not (0.5 <= a.cuda_overhead_gib < a.max_gpu_gib-0.5):
        p.error('Invalid GPU budget; require 3 <= max-gpu-gib <= 80 and 0.5 <= overhead < budget-0.5')
    if a.updates_per_epoch*a.grad_accum % (12//a.batch_size) != 0:
        p.error('updates-per-epoch * grad-accum must cover whole six-emotion sampling cycles')
    if a.window_seconds < 1.0 or a.window_seconds > 8.0:
        p.error('Window seconds must be in [1, 8]')
    if not (0 < a.eval_hop_fraction <= 1):
        p.error('Hop fraction must be in (0, 1]')
    if a.max_eval_windows < 1 or a.eval_batch_size < 1 or (a.inference_batch_size is not None and a.inference_batch_size < 1) or a.num_workers < 0 or a.eval_workers < 0:
        p.error('Invalid evaluation/worker configuration')
    if a.max_epochs <= a.frozen_epochs or a.frozen_epochs < 1 or a.patience < 1:
        p.error('Must have 1 <= frozen-epochs < max-epochs and patience >= 1')
    if a.updates_per_epoch < 0 or a.bootstrap_repeats < 100:
        p.error('Need >=0 optimizer updates (0=auto) and >=100 bootstrap replicates')
    if a.mode in ('all', 'test') and not a.unlock_test:
        p.error('Final per-speaker recording test is locked. Supply --unlock-test explicitly.')
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
    log = logging.getLogger('WAVLM_BASELINE_WITHIN_CORPUS')
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
    selected={}
    for c in CORPORA:
        sub=df[df.corpus==c]
        if sub.speaker_id.nunique()<2:
            viability_errors.append(f'{c}: fewer than 2 speakers after audit')
        missing=[LABELS[k] for k in range(NC) if int((sub.emotion_id==k).sum())<1]
        if missing:
            viability_errors.append(f'{c}: absent emotion classes after quarantine: {missing}')
        if sub.speaker_id.nunique()>=2 and not missing:
            try:
                selected[c]=choose_within_corpus_recordings(sub,a.seed)
                partitions=partition_experiment(df,selected,c)
                log.info('[RECORDING SPLIT] %-8s speakers in EACH split=%d counts=%s fractions=%s',
                    c, sub.speaker_id.nunique(), selected[c]['recordings'], selected[c]['fractions'])
            except (RuntimeError,ValueError,AssertionError) as exc:
                viability_errors.append(f'{c}: within-speaker allocation failed: {exc}')

    manifest_rows=[];counts=[];speaker_counts=[]
    for corpus,meta in selected.items():
        corpus_df=df[df.corpus==corpus]
        assignment={rid:split for split in ('train','val','test') for rid in meta[split]}
        assert len(assignment)==len(corpus_df) and set(assignment)==set(corpus_df.recording_id)
        for _,record in corpus_df.iterrows():
            manifest_rows.append({'corpus':corpus,'recording_id':record.recording_id,
                'speaker_id':record.speaker_id,'emotion_id':int(record.emotion_id),
                'emotion':record.emotion,'split':assignment[record.recording_id],
                'sha256':record.sha256,'pcm_sha256':record.pcm_sha256,'rel_path':record.rel_path})
        for split in ('train','val','test'):
            subset=corpus_df[corpus_df.recording_id.isin(meta[split])]
            counts.append({'corpus':corpus,'split':split,'n_recordings':len(subset),
                'n_speakers':int(subset.speaker_id.nunique()),
                'recording_fraction':float(meta['fractions'][split]),
                **{f'n_{SLUGS[i]}':int((subset.emotion_id==i).sum()) for i in range(NC)}})
        for v in meta['per_speaker']:
            speaker_counts.append({'corpus':corpus,**v})
    pd.DataFrame(manifest_rows, columns=['corpus','recording_id','speaker_id','emotion_id','emotion','split','sha256','pcm_sha256','rel_path']).sort_values(['corpus','recording_id']).to_csv(
        audit_dir/'recording_split_manifest.csv',index=False)
    pd.DataFrame(counts).to_csv(audit_dir/'split_summary.csv',index=False)
    pd.DataFrame(speaker_counts).to_csv(audit_dir/'per_speaker_split_counts.csv',index=False)
    log.info('[SPLIT AUDIT] Every speaker occurs in all three folds; original WAVs/hashes disjoint.')

    fingerprint = hashlib.sha256(df[['recording_id', 'speaker_id', 'emotion_id',
          'rel_path', 'orig_frames', 'orig_sr', 'sha256', 'pcm_sha256']].to_csv(index=False).encode()).hexdigest()
    split_metadata = {
        'seed': a.seed, 'split_policy': 'recording-disjoint, per-speaker 70/15/15 (all speakers in train, val, test), 6 classes',
        'recording_partitions': selected, 'labels': LABELS, 'audit_policy_version': AUDIT_POLICY_VERSION,
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
        'raw_data_unchanged': True, 'test_recordings_never_used_for_checkpoint_selection': True,
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
    review_lines.extend(['', '## Per-speaker recording-disjoint train/val/test viability', ''])
    review_lines.extend(f'- **BLOCKER**: {e}' for e in viability_errors)
    if not viability_errors:
        review_lines.append('- Pass: every speaker has recordings in each partition; all recordings are disjoint and all six emotions are covered.')
    (audit_dir/'AUDIT_REVIEW.md').write_text('\n'.join(review_lines)+'\n', encoding='utf-8')

    if viability_errors:
        raise RuntimeError(f'Audit scanned all corpora and saved diagnostics, but found '
            f'{len(viability_errors)} dataset viability issue(s): {viability_errors}. '
            f'Read {audit_dir/"AUDIT_REVIEW.md"}.')
    immutable = audit_dir/'immutable_per_speaker_recording_split.json'
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


def choose_within_corpus_recordings(frame, seed):
    """Deterministic approximate 70/15/15 *within each speaker*, recording-disjoint.

    All speakers appear in train, validation and test. Emotion-stratified allocation
    respects exact integer per-speaker totals, minimizing per-emotion proportion errors.
    For classes with >=3 items we try to guarantee >=1 example in every split,
    subject to per-speaker size constraints. Rare classes are not invented.
    """
    if frame.empty:
        raise RuntimeError('No recordings to split')
    corpus = str(frame.corpus.iloc[0])
    if frame.recording_id.duplicated().any():
        raise RuntimeError(f'{corpus}: duplicate recording identity reached split allocation')
    result = {'train': [], 'val': [], 'test': []}
    per_speaker = []
    frac = np.array([.70, .15, .15], dtype=np.float64)
    parts = ('train','val','test')
    for speaker, grp in frame.groupby('speaker_id', sort=True):
        n = len(grp)
        if n < 3:
            raise RuntimeError(f'{corpus}: {speaker} has only {n} recordings; cannot occupy three splits')
        ntrain = max(1, int(round(.70*n)))
        nval = max(1, int(round(.15*n)))
        ntest = n-ntrain-nval
        if ntest < 1:
            ntrain -= (1-ntest)
            ntest = 1
        targets = np.array([ntrain,nval,ntest], dtype=np.int64)
        if (targets < 1).any() or targets.sum()!=n:
            raise RuntimeError(f'Bad quotas for {speaker}: {targets.tolist()} / {n}')
        groups = {c:sorted(grp.loc[grp.emotion_id==c, 'recording_id'].tolist())
                  for c in range(NC)}
        totals = np.array([len(groups[c]) for c in range(NC)],dtype=np.int64)
        alloc = np.zeros((NC, 3),dtype=np.int64)
        remaining = targets.copy()
        # Prioritize common classes for all three partitions. This never draws
        # from other speakers and never splits a recording/window across folds.
        classes = sorted(range(NC), key=lambda c:(-int(totals[c]),
             stable_seed(42,corpus,speaker,c,'priority')))
        for c in classes:
            if totals[c] >= 3 and (remaining > 0).all():
                alloc[c] += (1,1,1)
                remaining -= 1
            elif totals[c] > 0 and remaining[0] > 0:
                alloc[c,0] += 1
                remaining[0] -= 1
        leftover = totals-alloc.sum(axis=1)
        if (leftover < 0).any():
            raise RuntimeError(f'{speaker}: minimum allocation exceeded class count')
        if int(leftover.sum()) != int(remaining.sum()):
            raise RuntimeError('Internal split capacity mismatch')
        # Greedy minimization of squared per-class ratio deviations, with
        # deterministic tie-breaking. Exact per-speaker quotas are constraints.
        for _ in range(int(leftover.sum())):
            choices=[]
            for c in range(NC):
                if leftover[c] <= 0:
                    continue
                for p in range(3):
                    if remaining[p] <= 0:
                        continue
                    ideal=float(totals[c])*float(frac[p])
                    old=float(alloc[c,p])-ideal
                    delta=((old+1)**2-old**2)/max(1,int(totals[c]))
                    tie=stable_seed(seed,corpus,speaker,c,p,int(alloc[c,p]),'greedy')
                    choices.append((round(delta,12),tie,c,p))
            if not choices:
                raise RuntimeError(f'{speaker}: cannot satisfy speaker split quotas')
            _,_,c,p = min(choices)
            alloc[c,p] += 1
            leftover[c] -= 1
            remaining[p] -= 1
        if (remaining != 0).any() or not np.array_equal(alloc.sum(axis=1),totals):
            raise RuntimeError(f'{speaker}: split apportionment failed')
        # Shuffle filenames independently per class; split allocation depends
        # on annotations/seed only and is identical for both model scripts.
        for c in range(NC):
            ids=list(groups[c])
            rng=np.random.default_rng(stable_seed(seed, corpus, speaker,c,'within_spk_70_15_15'))
            rng.shuffle(ids)
            offset=0
            for p, name in enumerate(parts):
                take=int(alloc[c,p])
                result[name].extend(ids[offset:offset+take])
                offset += take
            assert offset==len(ids)
        per_speaker.append({'speaker_id':speaker, 'total':n,
              'train':int(targets[0]),'val':int(targets[1]),'test':int(targets[2]),
              'train_frac':float(targets[0]/n),'val_frac':float(targets[1]/n),
              'test_frac':float(targets[2]/n)})
    for p in parts:
        result[p]=sorted(result[p])
    if len(set(result['train'])|set(result['val'])|set(result['test'])) != len(frame):
        raise RuntimeError('Missing or overlapping recording IDs in within-speaker partition')
    if any(set(result[x]) & set(result[y]) for x,y in
           (('train','val'),('train','test'),('val','test'))):
        raise RuntimeError('A recording appears in two splits')
    allids=set(frame.recording_id)
    if set().union(*(set(result[p]) for p in parts)) != allids:
        raise RuntimeError('Split coverage failed')
    result['recordings']={p:len(result[p]) for p in parts}
    result['fractions']={p:round(len(result[p])/len(frame),6) for p in parts}
    result['speakers']={p:int(frame.speaker_id.nunique()) for p in parts}
    result['per_speaker']=per_speaker
    return result


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


def partition_experiment(df, split_map, dataset):
    frame=df[df.corpus==dataset].copy()
    if frame.empty:
        raise RuntimeError(f'{dataset}: no retained recordings')
    selected=split_map[dataset]
    splits=tuple(frame.loc[frame.recording_id.isin(selected[sp])].copy()
                 for sp in ('train','val','test'))
    train,val,test=splits
    assert sum(map(len,splits))==len(frame)
    speakers=set(frame.speaker_id)
    # EXACTLY the user's protocol: EVERY speaker has recordings in EVERY split.
    assert all(set(part.speaker_id)==speakers for part in splits), 'Speaker missing from a partition'
    assert all(set(part.emotion_id)==set(range(NC)) for part in splits), 'Missing emotion in a partition'
    for a,b in ((train,val),(train,test),(val,test)):
        assert set(a.recording_id).isdisjoint(b.recording_id)
        assert set(a.sha256).isdisjoint(b.sha256)
        assert set(a.pcm_sha256).isdisjoint(b.pcm_sha256)
    for speaker in sorted(speakers):
        n=sum((part.speaker_id==speaker).sum() for part in splits)
        ntr=max(1,round(.7*n));nva=max(1,round(.15*n))
        nte=n-ntr-nva
        if nte<1:ntr-=1-nte;nte=1
        actual=[int((p.speaker_id==speaker).sum()) for p in splits]
        assert actual==[ntr,nva,nte], f'{speaker}: per-speaker 70/15/15 mismatch {actual}'
    if train.speaker_id.nunique()<2:
        raise RuntimeError('Contrastive pairs require two or more training speakers')
    return splits

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
    """Same-emotion positive pairs always come from DIFFERENT speakers.

    Balanced six-class exposure, no corpus mixing and no validation/test samples.
    Each class/speaker bucket is traversed without replacement until exhausted,
    then reshuffled to support naturally small datasets. The reproducible sampler
    is shared by Architecture A and the plain WavLM baseline.
    """
    def __init__(self, ds, batch_size, n_batches, seed):
        self.ds, self.batch_size, self.n_batches, self.seed = ds, batch_size, n_batches, seed
        self.epoch = 0
        self.groups = defaultdict(lambda: defaultdict(list))
        for i, row in enumerate(ds.rows):
            self.groups[int(row['emotion_id'])][row['speaker_id']].append(i)
        for cls in range(NC):
            if len(self.groups[cls]) < 2:
                raise RuntimeError(f'Class {cls} needs two distinct training speakers')
        if batch_size not in (2,6,12) or n_batches % (12//batch_size):
            raise ValueError('Microbatch must be 2, 6 or 12 and complete six-class cycles')

    def __len__(self):
        return self.n_batches

    def __iter__(self):
        rng = np.random.default_rng(stable_seed(self.seed, self.epoch, 'within_corpus_sampler_v1'))
        streams = { (c,s): rng.permutation(idxs).tolist()
                    for c, groups in self.groups.items() for s, idxs in groups.items() }
        offsets = {key: 0 for key in streams}
        def draw(c, s):
            key = (c,s)
            if offsets[key] >= len(streams[key]):
                streams[key] = rng.permutation(self.groups[c][s]).tolist()
                offsets[key] = 0
            i = int(streams[key][offsets[key]])
            offsets[key] += 1
            return i
        for batch_index in range(self.n_batches):
            group, pos = divmod(batch_index, NC // (self.batch_size//2))
            if pos == 0:
                order = rng.permutation(NC)
            cls_ids = order[pos*(self.batch_size//2):(pos+1)*(self.batch_size//2)]
            batch = []
            for c in cls_ids:
                c = int(c)
                speakers = sorted(self.groups[c])
                a,b = rng.choice(len(speakers), 2, replace=False)
                batch.extend((draw(c, speakers[int(a)]), draw(c, speakers[int(b)])))
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
    if embedding_matrix.shape != (len(preds), 1024):
        raise RuntimeError('Incorrect 1024-D WavLM baseline embedding matrix shape')
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


# Baseline model: no intermediate hidden states, no GHF, no DPTP, no SupCon.
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader, Sampler


def conv_frame_lengths(input_len, kernels, strides):
    lengths = input_len.long()
    for kernel, stride in zip(kernels, strides):
        lengths = torch.div(lengths-int(kernel), int(stride), rounding_mode='floor')+1
    return lengths



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
        raise RuntimeError('Completed test receipt exists; recovery forbidden')
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


class MaskedMeanPool(nn.Module):
    """True feature-frame mask, not a mean over padded zero/normalized frames."""
    def forward(self, sequence, frame_mask):
        if frame_mask.dtype != torch.bool:
            frame_mask = frame_mask.bool()
        if sequence.ndim != 3 or frame_mask.shape != sequence.shape[:2]:
            raise ValueError('Wrong sequence/mask dimensions for mean pooling')
        counts = frame_mask.sum(dim=1, keepdim=True)
        if (counts == 0).any():
            raise RuntimeError('All-padded audio encountered')
        # Accumulate in FP32 under BF16 autocast for numerically stable pooling.
        pooled = (sequence.float()*frame_mask.unsqueeze(-1)).sum(dim=1)/counts.float()
        return pooled.to(sequence.dtype)


class WavLMLargeBaseline(nn.Module):
    """Pretrained WavLM-Large -> last hidden state -> masked mean -> simple linear head."""
    def __init__(self, wavlm):
        super().__init__()
        self.wavlm = wavlm
        self.pool = MaskedMeanPool()
        self.norm = nn.LayerNorm(1024)
        self.dropout = nn.Dropout(0.30)
        self.classifier = nn.Linear(1024, NC)
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

    def forward(self, input_values, attention_mask):
        context = torch.no_grad() if self.stage == 'frozen' else contextlib.nullcontext()
        with context:
            output = self.wavlm(input_values, attention_mask=attention_mask,
                                output_hidden_states=False, return_dict=True)
        sequence = output.last_hidden_state
        lengths = conv_frame_lengths(attention_mask.long().sum(-1),
                                     self.wavlm.config.conv_kernel,
                                     self.wavlm.config.conv_stride)
        if torch.any(lengths <= 0) or torch.any(lengths > sequence.shape[1]):
            raise RuntimeError('Invalid convolutional feature-frame lengths')
        frame_mask = torch.arange(sequence.shape[1], device=sequence.device)[None, :] < lengths[:, None]
        pooled = self.pool(sequence, frame_mask)
        z = self.norm(pooled)
        logits = self.classifier(self.dropout(z))
        return {'logits': logits, 'embedding': z}

def configure_cuda_budget(max_gpu_gib, overhead_gib, log):
    """Fail closed at the caching-allocator cap; NOT an NVML/MIG total-VRAM quota.

    Non-PyTorch CUDA allocations and the driver context are excluded, hence the
    configured headroom. An actual hard physical limit needs GPU partitioning
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
    log.info('[VRAM BUDGET] Process target <= %.2f GiB; PyTorch caching allocator capped '
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

    axs[1, 0].plot(
        df.epoch,
        df['val_macro_corpus_uar'] * 100,
        label=f'{heldout.upper()} Validation UAR',
        linewidth=2
    )
    
    axs[1, 0].plot(
        df.epoch,
        df['val_macro_corpus_f1'] * 100,
        label=f'{heldout.upper()} Validation Macro-F1',
        linewidth=2
    )
    
    axs[1, 0].set_ylabel('Validation Metrics (%)')
    axs[1, 0].set_ylim(0, 105)
    axs[1, 0].legend()
    
    axs[1, 1].plot(df.epoch, df.peak_gpu_gib, label='peak GPU GiB')
    axs[1, 1].set_ylabel('GPU GiB')
    axs[1, 1].legend()
    for ax in axs.flat:
        ax.set_xlabel('Epoch')
        ax.grid(alpha=.2)
        ax.axvline(df.loc[df.stage == 'finetune', 'epoch'].min()-.5,
                   color='gray', linestyle='--', linewidth=1)
    fig.suptitle(f'WavLM-Large baseline Within-corpus — {heldout.upper()} (speaker-overlapping recording-disjoint validation)')
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
        self.output = a.output_dir/('dataset_'+heldout.replace('-', '_'))
        self.output.mkdir(parents=True, exist_ok=True)
        self.store = AudioStore(a.output_dir/'shared_16khz_audio_cache', a.cache_audio)
        self.window = round(a.window_seconds*SAMPLE_RATE)
        self.device = torch.device('cuda:0')
        self.split_fingerprint = hashlib.sha256((immutable['manifest_fingerprint']+'|'+heldout+'|'+
            self.train_df.recording_id.sort_values().to_csv(index=False)+'|'+
            self.val_df.recording_id.sort_values().to_csv(index=False)+'|'+
            self.test_df.recording_id.sort_values().to_csv(index=False)).encode()).hexdigest()
        self.config = {'protocol': PROTOCOL, 'dataset': heldout,
                       'training_corpora': [heldout],
                       'within_corpus': True, 'speaker_overlap_all_partitions': True, 'recording_disjoint_all_partitions': True,
                       'six_class_ontology': LABELS,
                       'pretrained': PRETRAINED_MODEL, 'checkpoint_revision': a.checkpoint_revision,
                       'model_architecture': 'WavLM-Large [24x1024] -> last-layer masked mean pool -> LayerNorm(1024) -> Dropout(.30) -> Linear(1024,6)',
                       'learning_stages': f'{a.frozen_epochs} frozen + up to {a.max_epochs-a.frozen_epochs} top-6-layer finetuning',
                       'audio': {'resample_hz': SAMPLE_RATE, 'crop_seconds': a.window_seconds,
                                 'window_samples': self.window, 'train_crop': '5-region cyclic with +/-300ms jitter',
                                 'eval_sliding_hop_fraction': a.eval_hop_fraction,
                                 'max_eval_windows': a.max_eval_windows,
                                 'amplitude': 'no independent peak/loudness normalization; HF masked standardization',
                                 'silence_trimming': 'none; retain emotion-related pauses'},
                       'augmentation': '15% low-level Gaussian noise (23-36 dB SNR), training only',
                       'batch': {'micro': a.batch_size, 'grad_accum': a.grad_accum,
                                 'effective': a.batch_size*a.grad_accum,
                                 'updates_per_virtual_epoch': a.updates_per_epoch,
                                 'balanced_classes': True, 'distinct_speaker_same_emotion_pairs': True,
                                 'class_balanced_within_corpus': True},
                       'gpu_budget': {'target_gib': a.max_gpu_gib,
                                      'pytorch_allocator_gib': a.max_gpu_gib-a.cuda_overhead_gib,
                                      'non_pytorch_headroom_gib': a.cuda_overhead_gib},
                       'objective': 'CrossEntropy(label_smoothing=0.05) only; no SupCon',
                       'checkpoint_selection': 'within-corpus validation UAR; validation recordings disjoint from train/test, speakers overlap',
                       'early_stopping': {'patience': a.patience, 'min_delta': a.min_delta},
                       'optimizer_schedule': {'head_lr_frozen': 3e-4, 'head_lr_finetune': 1e-4,
                           'top6_lr': 5e-6, 'layerwise_decay': 0.8, 'weight_decay': 0.01,
                           'warmup_fraction_per_stage': 0.10, 'cosine_decay_floor': 0.05,
                           'max_epochs': a.max_epochs},
                       'seed': a.seed, 'recording_partitions': {k: val_map[heldout][k] for k in ('train','val','test')},
                       'recording_fractions': val_map[heldout]['fractions'],
                       'split_fingerprint': self.split_fingerprint,
                       'manifest_fingerprint': immutable['manifest_fingerprint'],
                       'sizes': {'train': len(self.train_df), 'val': len(self.val_df), 'test': len(self.test_df)},
                       'test_speaker_count': int(self.test_df.speaker_id.nunique()),
                       'split_semantics': 'each speaker stratified into recording-disjoint ~70/15/15 train/validation/test',
                       'python': platform.python_version(), 'pytorch': torch.__version__}
        old = self.output/'resolved_config.json'
        # Re-running with modified training/preprocessing settings must not silently
        # load an incompatible checkpoint or append to a different experiment.
        if old.exists():
            cfg = json.loads(old.read_text())
            for key in ('split_fingerprint', 'checkpoint_revision', 'seed', 'batch', 'gpu_budget',
                        'audio', 'learning_stages', 'objective',
                        'early_stopping', 'optimizer_schedule'):
                if cfg[key] != self.config[key]:
                    raise RuntimeError(f'{heldout} existing outputs use different {key}; choose NEW --output-dir')
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
        self.model = WavLMLargeBaseline(backbone).to(self.device)
        self.model.set_stage('frozen')
        try:
            self.model.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        except TypeError:
            self.model.wavlm.gradient_checkpointing_enable()
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
            decay = .01 if (param.ndim >= 2 and not name.endswith('bias')) else 0.
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
        self.sampler = CorpusCrossSpeakerPairSampler(
            self.train_ds, self.a.batch_size,
            self.a.updates_per_epoch*self.a.grad_accum, self.a.seed)
        # Verify each same-class positive pair comes from different TRAINING speakers.
        for idx, batch in enumerate(self.sampler):
            by_class = defaultdict(list)
            for i in batch:
                r = self.train_ds.rows[i]
                by_class[r['emotion_id']].append(r)
            assert all(len(group) == 2 and group[0]['speaker_id'] != group[1]['speaker_id']
                       for group in by_class.values())
            if idx > 12:
                break

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
        sums, counts = {}, Counter()
        embeddings = {} if collect_embeddings else None
        with torch.inference_mode():
            for batch in tqdm(loader, desc=f'[EVAL {self.heldout}] {phase}',
                              mininterval=2., disable=not sys.stderr.isatty()):
                with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                    outputs = self.model(batch['input_values'].to(self.device, non_blocking=True),
                                         batch['attention_mask'].to(self.device, non_blocking=True))
                logits = outputs['logits'].float().cpu().numpy()
                vectors = outputs['embedding'].float().cpu().numpy() if collect_embeddings else None
                if not np.isfinite(logits).all() or (vectors is not None and not np.isfinite(vectors).all()):
                    raise FloatingPointError('Non-finite baseline inference outputs')
                for i, rid in enumerate(batch['recording_id']):
                    if rid not in sums:
                        sums[rid] = np.zeros(NC, dtype=np.float64)
                        if collect_embeddings:
                            embeddings[rid] = np.zeros(1024, dtype=np.float64)
                    sums[rid] += logits[i]
                    if collect_embeddings:
                        embeddings[rid] += vectors[i]
                    counts[rid] += 1
        mem = gpu_memory_snapshot(self.device)
        if mem['process_nvidia_smi_gib'] is not None and mem['process_nvidia_smi_gib'] > self.a.max_gpu_gib:
            raise RuntimeError(f"Evaluation exceeded VRAM process target: {mem['process_nvidia_smi_gib']:.2f} GiB")
        if set(counts) != set(df.recording_id):
            raise RuntimeError('Missing/extra recording in baseline evaluation')
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
            rows.append(row)
        pred_frame = pd.DataFrame(rows).sort_values('recording_id').reset_index(drop=True)
        if collect_embeddings:
            mat = np.stack([(embeddings[rid]/counts[rid]).astype('float32')
                            for rid in pred_frame.recording_id])
            assert mat.shape == (len(pred_frame), 1024)
            return pred_frame, mat
        return pred_frame

    def train_epoch(self, epoch, opt, scheduler):
        self.model.train()
        self.train_ds.epoch = epoch
        self.sampler.epoch = epoch
        loader = DataLoader(self.train_ds, batch_sampler=self.sampler,
                            num_workers=self.a.num_workers, pin_memory=True,
                            collate_fn=self.collate,
                            **({'prefetch_factor': 2} if self.a.num_workers else {}))
        assert len(loader) == self.a.updates_per_epoch*self.a.grad_accum
        totals = Counter()
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
                out = self.model(input_values, attention_mask)
                loss = F.cross_entropy(out['logits'].float(), y, label_smoothing=.05)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite baseline loss at epoch={epoch} batch={bi}')
            (loss/self.a.grad_accum).backward()
            bs = len(y)
            totals['seen'] += bs
            totals['correct'] += int((out['logits'].detach().argmax(-1) == y).sum().item())
            totals['loss'] += float(loss.item())*bs
            if (bi+1)%self.a.grad_accum == 0:
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
                        raise RuntimeError('GPU process exceeded target: '
                                           f"{mem['process_nvidia_smi_gib']:.2f} > {self.a.max_gpu_gib:.2f} GiB. "
                                           'Reduce --batch-size or --max-gpu-gib, or isolate with MIG.')
                    self.log.info('[%s epoch=%d] steps=%d/%d, loss=%.4f, crop_acc=%.3f, '
                                  'GPU reserved=%.2f GiB peak=%.2f GiB, process=%s GiB',
                                  self.heldout, epoch, steps, self.a.updates_per_epoch,
                                  totals['loss']/totals['seen'], totals['correct']/totals['seen'],
                                  mem['reserved_gib'], mem['peak_reserved_gib'],
                                  ('n/a' if mem['process_nvidia_smi_gib'] is None else
                                   f"{mem['process_nvidia_smi_gib']:.2f}"))
        assert steps == self.a.updates_per_epoch
        return {'epoch': epoch, 'stage': self.model.stage,
                'train_loss': totals['loss']/totals['seen'],
                'train_ce': totals['loss']/totals['seen'],
                'train_accuracy': totals['correct']/totals['seen'],
                'n_crop_examples': int(totals['seen']), 'updates': steps,
                'seconds': time.perf_counter()-start,
                'peak_gpu_gib': torch.cuda.max_memory_allocated(self.device)/2**30,
                'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(self.device)/2**30}

    def validate(self, epoch):
        preds = self.infer_recordings(self.val_df, 'val', epoch)
        scores = quick_metrics(preds)
        return {'val_macro_corpus_uar':scores['uar'],
                'val_macro_corpus_f1':scores['macro_f1'],
                'val_macro_corpus_acc':scores['accuracy'],
                'val_loss':scores['cross_entropy']}, preds

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
            raise RuntimeError('Checkpoint belongs to a different dataset or speaker partition')
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
                last_stage = stage
                resume = None
            train_row = self.train_epoch(epoch, opt, sched)
            val_row, val_pred = self.validate(epoch)
            row = {**train_row, **val_row}
            history.append(row)
            pd.DataFrame(history).to_csv(self.output/'train_history.csv', index=False)
            plot_training(history, self.output, self.heldout)
            # Selection follows WITHIN VALIDATION ONLY. Test audio NEVER enters here.
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
                self.log.info('[%s] New best epoch=%d WITHIN VAL mean-UAR=%.4f',
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
        np.save(self.output/'validation'/'within_validation_embeddings_1024d.npy', val_emb, allow_pickle=False)
        val_pred[['recording_id', 'speaker_id', 'true_class', 'corpus']].to_csv(
            self.output/'validation'/'within_validation_embedding_ids.csv', index=False)
        # Within-corpus validation on non-overlapping recordings from the same speakers. Final test recordings remain locked.
        export_scores(val_pred, self.output/'validation'/self.heldout,
                      f'{self.heldout.upper()} within-corpus recording-disjoint speaker-overlapping validation',
                      self.a.bootstrap_repeats, self.a.seed)
        export_scores(val_pred, self.output/'validation'/'pooled_sources',
                      'Within-corpus validation, seen speakers / unseen recordings (NOT final test)',
                      self.a.bootstrap_repeats, self.a.seed)
        # Plain baseline has no trainable cross-layer fusion or gates.

        write_json(done, {'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                          'experiment': self.heldout, 'best_epoch': best_epoch,
                          'best_validation': best['validation'],
                          'best_checkpoint_sha256': digest_file(self.output/'best.pth'),
                          'split_fingerprint': self.split_fingerprint,
                          'note': 'Test recordings were not evaluated during checkpoint selection'})
        self.log.info('[%s] Training finished; best checkpoint epoch=%d', self.heldout, best_epoch)

    def test(self):
        if not self.a.unlock_test:
            raise PermissionError('Explicit --unlock-test required')
        done = self.output/'training_complete.json'
        if not done.exists():
            raise RuntimeError(f'{self.heldout}: training must finish before its per-speaker held-out-recording test')
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
                                f'{self.heldout.upper()} held-out recordings from seen speakers (within-corpus)',
                                self.a.bootstrap_repeats, self.a.seed)
        np.save(self.output/'test'/'recording_embeddings_1024d.npy', test_embeddings, allow_pickle=False)
        preds[['recording_id', 'speaker_id', 'true_class', 'corpus']].to_csv(
            self.output/'test'/'embedding_recording_ids.csv', index=False)
        plot_embedding_pca(test_embeddings, preds, self.output/'test'/'embedding_pca_2d.png',
                           f'{self.heldout.upper()} held-out recordings from seen speakers in the same corpus: 1024-D embedding PCA')
        metrics.update({'dataset': self.heldout, 'best_epoch': int(cp['epoch']),
                        'validation_uar': cp['validation']['val_macro_corpus_uar'],
                        'checkpoint_sha256': sha, 'split_fingerprint': self.split_fingerprint,
                        'test_protocol': 'within-corpus per-speaker 70/15/15, original recordings disjoint, speakers overlap, test once'})
        write_json(self.output/'test'/'metrics.json', metrics)
        write_json(receipt, {'executed_at_utc': datetime.now(timezone.utc).isoformat(),
                             'dataset': self.heldout, 'checkpoint_sha256': sha,
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
    """Publication outputs: one row per SAME-corpus recording-held-out, speaker-overlapping test."""
    report_dir = a.output_dir/'paper_results'
    report_dir.mkdir(parents=True, exist_ok=True)
    rows, class_rows, confusions, gate_rows = [], [], {}, []
    for dataset in CORPORA:
        parent = a.output_dir/('dataset_'+dataset.replace('-', '_'))
        receipt = parent/'final_test_receipt.json'
        path = parent/'test'
        met = path/'metrics.json'
        if not receipt.exists() or not met.exists():
            log.info('[REPORT] %s: final test not completed', dataset)
            continue
        m = json.loads(met.read_text())
        conf = pd.read_csv(path/'confusion_normalized.csv', index_col=0)
        cfg = json.loads((parent/'resolved_config.json').read_text())
        fractions = cfg['recording_fractions']
        rows.append({'dataset':dataset, 'architecture': ARCH_NAME,
                     'train_n':cfg['sizes']['train'], 'val_n':cfg['sizes']['val'],
                     'test_n':int(m['n_recordings']), 'test_speakers':int(m['n_speakers']),
                     'train_pct':100*fractions['train'], 'val_pct':100*fractions['val'],
                     'test_pct':100*fractions['test'],
                     'accuracy':m['accuracy'], 'uar':m['uar'], 'macro_f1':m['macro_f1'],
                     'weighted_f1':m['weighted_f1'], 'auroc_ovr':m['auroc_macro_ovr'],
                     'ece_15':m['ece_15_bins'], 'cross_entropy':m['cross_entropy'],
                     'accuracy_ci_low':m['accuracy_95ci_speaker_bootstrap'][0],
                     'accuracy_ci_high':m['accuracy_95ci_speaker_bootstrap'][1],
                     'uar_ci_low':m['uar_95ci_speaker_bootstrap'][0],
                     'uar_ci_high':m['uar_95ci_speaker_bootstrap'][1],
                     'best_epoch':m['best_epoch'], 'split_fingerprint':m['split_fingerprint']})
        cl = pd.read_csv(path/'per_class_metrics.csv')
        for _, r in cl.iterrows():
            class_rows.append({'dataset':dataset, 'emotion':r.emotion, 'precision':r.precision,
                               'recall':r.recall, 'f1':r.f1, 'support':r.support})
        confusions[dataset] = conf.to_numpy()
        if ARCH_NAME == 'architecture_a':
            pred = pd.read_csv(path/'recording_predictions.csv')
            for i in range(4):
                gate_rows.append({'dataset':dataset, 'group':i+1,
                                  'mean_gate':float(pred[f'ghf_gate_{i+1}'].mean())})
    if not rows:
        log.info('[REPORT] No completed final test yet; no scores fabricated')
        return
    t = pd.DataFrame(rows)
    tc = pd.DataFrame(class_rows)
    t.to_csv(report_dir/'table_1_individual_corpus_metrics.csv', index=False)
    tc.to_csv(report_dir/'table_2_per_emotion_metrics.csv', index=False)
    if gate_rows:
        pd.DataFrame(gate_rows).to_csv(report_dir/'table_3_ghf_gates.csv', index=False)
    with (report_dir/'table_1_individual_corpus_metrics.tex').open('w') as f:
        latex = t[['dataset', 'train_n', 'val_n', 'test_n', 'accuracy', 'uar', 'macro_f1']].copy()
        for col in ('accuracy','uar','macro_f1'):
            latex[col] = latex[col].map(lambda x: f'{x*100:.2f}\\%')
        f.write(latex.to_latex(index=False,escape=False,caption='Speaker-overlapping recording-disjoint within-corpus tests (six shared emotions).'))
    x=np.arange(len(t))
    fig,ax=plt.subplots(figsize=(10,5))
    for j,key in enumerate(('accuracy','uar','macro_f1')):
        ax.bar(x+(j-1)*.23,t[key].to_numpy()*100,.23,label=key)
    ax.set_xticks(x,[s.upper() for s in t.dataset])
    ax.set_ylabel('Percent');ax.set_ylim(0,105)
    ax.set_title(f'{ARCH_NAME}: individual-corpus speaker-overlapping recording-disjoint evaluation')
    ax.legend();ax.grid(axis='y',alpha=.25)
    fig.tight_layout();fig.savefig(report_dir/'individual_corpus_metrics.png',dpi=200);plt.close(fig)
    for field in ('recall','f1'):
        pivot=tc.pivot(index='dataset',columns='emotion',values=field).reindex(
            index=list(t.dataset),columns=LABELS)
        plot_heatmap(pivot.to_numpy(),LABELS,[c.upper() for c in pivot.index],
                     report_dir/f'individual_corpus_{field}_heatmap.png',
                     f'Speaker-overlapping recording-disjoint {field} per emotion', percentages=True)
    if len(confusions)==4:
        fig,axes=plt.subplots(2,2,figsize=(12,10),constrained_layout=True)
        for ax,c in zip(axes.flat,CORPORA):
            mat=confusions[c];im=ax.imshow(mat,vmin=0,vmax=1,cmap='Blues')
            for i in range(NC):
                for j in range(NC):
                    ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=7)
            ax.set_xticks(range(NC),LABELS,rotation=45,ha='right');ax.set_yticks(range(NC),LABELS)
            ax.set_title(c.upper());ax.set_xlabel('Predicted');ax.set_ylabel('Actual')
        fig.colorbar(im,ax=list(axes.flat),shrink=.7)
        fig.savefig(report_dir/'four_normalized_confusions.png',dpi=200);plt.close(fig)
    methods=['# Within-corpus per-speaker 70/15/15 (speaker-overlapping) benchmark','',
        f'Architecture: {ARCH_NAME}',
        'Label ontology: '+', '.join(LABELS),
        'All speakers occur in train, validation and test; only recordings are mutually disjoint.',
        'Per-speaker splits are 70/15/15 rounded to whole recordings; exact percentages are integer-limited.',
        'See per-corpus actual fractions, immutable manifests, and excluded audio audit.',
        'The test set is evaluated once after validation-only checkpoint selection.',
        'Confidence intervals are speaker-cluster bootstrap intervals (dependent on number of distinct speakers).',
        'Only one random split and seed are reported unless experiments are independently repeated.','',
        '## Completed tests', '',
        '| Corpus | Train / Val / Test | Accuracy | UAR | Macro F1 |',
        '|---|---|---:|---:|---:|']
    for _,r in t.iterrows():
        methods.append(f'| {r.dataset} | {r.train_n} / {r.val_n} / {r.test_n} | '
                       f'{r.accuracy:.2%} | {r.uar:.2%} | {r.macro_f1:.2%} |')
    (report_dir/'RESEARCH_REPORT.md').write_text('\n'.join(methods)+'\n')
    write_json(report_dir/'report_manifest.json',{'architecture':ARCH_NAME,
         'completed_datasets':list(t.dataset),'completed':len(t),'expected':4})
    log.info('[REPORT] %s: %d/4 within-corpus final tests reported', ARCH_NAME,len(t))

def run_self_test():
    """CPU checks: filename parsers, every-speaker coverage, recording disjointness, sampler, gradients."""
    assert parse_record('crema-d', Path('1001_DFA_ANG_XX.wav')) == ('1001','angry')
    assert parse_record('emodb', Path('03a01Fa.wav')) == ('03','happy')
    assert parse_record('ravdess', Path('03-01-03-01-02-01-07.wav')) == ('07','happy')
    assert parse_record('emovo', Path('/tmp/f1/tri-f1-b1.wav')) == ('f1','sad')
    from types import SimpleNamespace
    rows=[]
    for corpus in CORPORA:
        nsp={'ravdess':8,'crema-d':10,'emodb':10,'emovo':6}[corpus]
        for s in range(nsp):
            for cls in range(NC):
                for rep in range(8):
                    rows.append({'corpus':corpus,'speaker_id':f'{corpus}:{s:03d}',
                                 'emotion_id':cls,'recording_id':f'{corpus}:{s}:{cls}:{rep}',
                                 'sha256':f'{corpus}:{s}:{cls}:{rep}',
                                 'pcm_sha256':f'{corpus}:{s}:{cls}:{rep}'})
    frame=pd.DataFrame(rows)
    maps={}
    for corpus in CORPORA:
        maps[corpus]=choose_within_corpus_recordings(frame[frame.corpus==corpus],42)
        tr,va,te=partition_experiment(frame,maps,corpus)
        assert len(tr)+len(va)+len(te)==len(frame[frame.corpus==corpus])
        all_speakers=set(frame.loc[frame.corpus==corpus,'speaker_id'])
        assert set(tr.speaker_id)==set(va.speaker_id)==set(te.speaker_id)==all_speakers
        assert not set(tr.recording_id)&set(va.recording_id)
        assert not set(tr.recording_id)&set(te.recording_id)
        assert not set(va.recording_id)&set(te.recording_id)
        ds=SimpleNamespace(rows=tr.to_dict('records'))
        for batch_size in (2,6,12):
            sampler=CorpusCrossSpeakerPairSampler(ds,batch_size,12//batch_size*3,42)
            for b in sampler:
                by_class=defaultdict(set)
                for idx in b:
                    r=ds.rows[idx];by_class[int(r['emotion_id'])].add(r['speaker_id'])
                assert all(len(v)==2 for v in by_class.values())
    assert stable_seed('same') == stable_seed('same')
    assert len(eval_slices(3*SAMPLE_RATE,3*SAMPLE_RATE))==1
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.manual_seed(42)
    head=nn.Sequential(nn.LayerNorm(1024),nn.Dropout(.3),nn.Linear(1024,NC))
    features=torch.randn(2,1024)
    F.cross_entropy(head(features),torch.tensor([0,1])).backward()
    assert head[-1].weight.grad is not None
    if ARCH_NAME == 'architecture_a':
        ghf=GroupedHierarchicalFusion(64)
        hs=tuple(torch.randn(2,8,64) for _ in range(24))
        mask=torch.tensor([[1]*8,[1]*5+[0]*3],dtype=torch.bool)
        fused,w,g=ghf(hs,mask)
        assert fused.shape==(2,8,64)
        assert DualPathTemporalPooling(64,8)(fused,mask).shape==(2,512)
    else:
        v=torch.randn(2,7,1024)
        m=torch.tensor([[1]*7,[1]*4+[0]*3],dtype=torch.bool)
        assert MaskedMeanPool()(v,m).shape==(2,1024)
    print('[SELF-TEST PASSED] parsers, six-class ontology, per-speaker 70/15/15 recording splits, '
          'same-class distinct-speaker pairs, pooling/classifier gradient')

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
        write_comparison_report(a, log)
        return
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
        log.info('WITHIN-CORPUS %s: ALL speakers in train/val/test, disjoint recordings', heldout)
        from copy import copy
        experiment_args = copy(a)
        if experiment_args.updates_per_epoch == 0:
            train_df, _, _ = partition_experiment(df, val_speakers, heldout)
            experiment_args.updates_per_epoch = math.ceil(len(train_df)/24)
        log.info('[EPOCH SIZE] %s: optimizer updates=%d, effective batch=24',
                 heldout, experiment_args.updates_per_epoch)
        ex = Experiment(experiment_args, df, val_speakers, immutable, heldout, log)
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
        print('[CUDA LIMIT] The low-VRAM allocator refused an allocation rather than exceeding its cap. '
              'Try --window-seconds 2.0 with a NEW --output-dir; do not increase batch size. '
              'The 9 GiB total target is a conservative estimate, not a hardware VRAM partition.',
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
