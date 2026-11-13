#!/usr/bin/env python3
"""EAV Speech V2 Fast — dataset-adapted PME4 Architecture A, 42 subjects, five classes.

Keep WavLM-Large + 24-layer GHF + DPTP(512) + SupCon architecture unchanged.
Default evaluation uses 4×5-second crops spanning the full ~20-second EAV
speaking recording; only RECORDINGS are split, never individual windows.

Research protocol: per subject × emotion 14 train / 3 validation / 3 test
recordings, all 42 subjects in every split (subject-dependent, not LOSO).
The test remains locked until a single explicit final evaluation.
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, random, sys, time, traceback, datetime, warnings
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
from scipy.signal import resample_poly
from scipy.special import softmax
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader,Dataset,Sampler
from tqdm.auto import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from eav_common import (LABELS,SLUGS,PROTOCOL,DEFAULT_DATA,DEFAULT_OUTPUT,
                        audit_or_load,manifest_sha,json_dump,report_predictions,sha_file)

RATE=16000
FORMAT='EAV_ARCH_A_WAVLM_LARGE_GHF_DPTP_SUPCON_5CLASS_V2'
MODEL_NAME='microsoft/wavlm-large'
PINNED_REVISION='c1423ed94bb01d80a3f5ce5bc39f6026a0f4828c'
DEFAULTS={'window_seconds':5.0,'eval_windows':0,'train_views_per_recording':1,
          'noise_probability':0.10,'head_lr':3e-4,'finetune_head_lr':1e-4,
          'wavlm_lr':5e-6,'supcon_weight':0.05,'label_smoothing':0.03,
          'frozen_epochs':4,'max_epochs':30,'patience':8,'seed':42,
          'batch_size':4,'grad_accum':5,'checkpoint_revision':PINNED_REVISION,
          'disable_supcon':False}
LOCKED_KEYS=tuple(DEFAULTS)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)


def stable_seed(*parts):
    return int.from_bytes(hashlib.sha256('|'.join(map(str,parts)).encode('utf8')).digest()[:8], 'little')


def rng_capture():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch_cpu':torch.get_rng_state(),
            'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def rng_restore(s):
    random.setstate(s['python']);np.random.set_state(s['numpy'])
    torch.set_rng_state(s['torch_cpu'])
    if torch.cuda.is_available() and s.get('torch_cuda'):torch.cuda.set_rng_state_all(s['torch_cuda'])


def fingerprint(d):
    return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def window_samples(args):
    return int(round(RATE*args.window_seconds))


def window_starts(num_samples, width, n_windows=0):
    """Cover the full waveform without cross-recording leakage.

    Automatic mode uses ceil(recording_duration / window_length) views. Evenly
    spaced overlapping starts ensure no portion is omitted, including audio
    whose resampled length is not exactly 20 seconds.
    """
    if num_samples<1600:raise ValueError(f'Short EAV waveform: {num_samples} samples')
    width=int(width)
    if width<=0:raise ValueError('Window samples must be positive')
    if num_samples<=width:return [0]
    k=int(n_windows) if n_windows else math.ceil(num_samples/width)
    if k<1:raise ValueError('Need >=1 evaluation window')
    if k==1:
        return [int(round((num_samples-width)/2))]  # centered quick-validation view
    starts=np.linspace(0,num_samples-width,k).round().astype(int).tolist()
    return starts


class AudioStore:
    """Versioned, atomically persisted 16 kHz mono resampled waveforms."""
    def __init__(self,root,cache=True):
        self.root=Path(root);self.root.mkdir(parents=True,exist_ok=True);self.cache=cache
    def get(self,row):
        raw=Path(row['audio_path'])
        target=self.root/f'{row["recording_id"]}_{row["audio_bytes"]}_{row["audio_mtime_ns"]}_16k.npy'
        if self.cache and target.exists():
            a=np.load(target,mmap_mode='r',allow_pickle=False)
            if a.ndim!=1 or len(a)<1600 or not np.isfinite(a).all():raise ValueError(f'Corrupt audio cache {target}')
            return a
        w,sr=sf.read(raw,dtype='float32',always_2d=True)
        if sr<=0 or w.shape[0]<1000 or w.shape[1]<1:raise ValueError(f'Invalid EAV WAV: {raw}')
        mono=w.mean(axis=1,dtype=np.float32)
        if sr!=RATE:
            gcd=math.gcd(int(sr),RATE)
            mono=resample_poly(mono,RATE//gcd,int(sr)//gcd).astype('float32')
        if not np.isfinite(mono).all() or np.max(np.abs(mono))<1e-9:
            raise ValueError(f'Silent / nonfinite EAV WAV: {raw}')
        seconds=len(mono)/RATE
        if not 5<=seconds<=30:raise ValueError(f'Unexpected EAV WAV duration {seconds:.2f}s: {raw}')
        if self.cache:
            temp=target.with_name(target.name+f'.{os.getpid()}.tmp')
            with open(temp,'wb') as fh:np.save(fh,mono,allow_pickle=False)
            os.replace(temp,target)
            return np.load(target,mmap_mode='r',allow_pickle=False)
        return mono


def choose_training_start(row,n,width,epoch,view_index,seed):
    """Epoch-rotating full-recording coverage, deterministic per original trial.

    Each epoch visits distinct temporal grid regions. Jitter <=0.20 seconds
    does not change the original recording split or emotion label.
    """
    positions=window_starts(n,width,0)
    region=(epoch+view_index+stable_seed(row['recording_id'],'region')%len(positions))%len(positions)
    anchor=positions[region]
    rng=np.random.default_rng(stable_seed(seed,epoch,view_index,row['recording_id'],'speech'))
    jitter=int(round(rng.uniform(-.2,.2)*RATE))
    return int(np.clip(anchor+jitter,0,max(0,n-width))),rng


def crop_train(row,wave,window,epoch,view_index,seed,noise_probability):
    a,rng=choose_training_start(row,len(wave),window,epoch,view_index,seed)
    x=np.asarray(wave[a:a+window],dtype=np.float32).copy()
    if len(x)<window:x=np.pad(x,(0,window-len(x)))
    if rng.random()<noise_probability:
        rms=float(np.sqrt(np.mean(x.astype('float64')**2)))
        if rms>1e-6:
            snr_db=rng.uniform(25,40)
            x=np.clip(x+rng.normal(0,rms/(10**(snr_db/20)),len(x)).astype('float32'),-1,1)
    return x


class TrainDataset(Dataset):
    def __init__(self,frame,store,window,seed=42,n_views=1,noise_probability=.1):
        self.records=frame.sort_values('recording_id').to_dict('records')
        self.rows=[dict(r,view_index=j) for r in self.records for j in range(n_views)]
        self.store=store;self.window=window;self.seed=seed
        self.noise_probability=noise_probability;self.epoch=0
    def __len__(self):return len(self.rows)
    def __getitem__(self,i):
        row=self.rows[i]
        return {'wave':crop_train(row,self.store.get(row),self.window,self.epoch,
                    row['view_index'],self.seed,self.noise_probability),
                'y':int(row['emotion_id']),'subject_id':int(row['subject_id']),
                'recording_id':row['recording_id']}


class EvalDataset(Dataset):
    def __init__(self,frame,store,window,eval_windows=0):
        self.rows=frame.sort_values('recording_id').to_dict('records')
        self.store=store;self.window=window
        self.items=[]
        for i,row in enumerate(self.rows):
            n=len(store.get(row))
            for start in window_starts(n,window,eval_windows):self.items.append((i,start))
    def __len__(self):return len(self.items)
    def __getitem__(self,i):
        ridx,start=self.items[i];row=self.rows[ridx]
        w=self.store.get(row)
        x=np.asarray(w[start:start+self.window],dtype=np.float32).copy()
        if len(x)<self.window:x=np.pad(x,(0,self.window-len(x)))
        return {'wave':x,'y':int(row['emotion_id']),
                'recording_id':row['recording_id'],'subject_id':int(row['subject_id'])}


def collate(extractor,window):
    def apply(items):
        feat=extractor([x['wave'] for x in items],sampling_rate=RATE,
                       return_tensors='pt',padding='max_length',max_length=window,
                       truncation=True,return_attention_mask=True)
        if (feat['attention_mask'].sum(-1)<1000).any():raise RuntimeError('Invalid WavLM audio mask')
        return {'input_values':feat['input_values'],'attention_mask':feat['attention_mask'].long(),
                'y':torch.tensor([x['y'] for x in items],dtype=torch.long),
                'recording_id':[x['recording_id'] for x in items]}
    return apply


class BalancedClassPairSampler(Sampler):
    """Each microbatch contains 2 same-class examples from DISTINCT subjects.

    For B=4 use two emotions per microbatch, with a five-edge cyclic schedule
    giving equal sample counts per emotion. B=10 covers all five emotions.
    This supplies genuine SupCon positives (not repeated windows of one record).
    """
    def __init__(self,ds,seed=42,batch_size=4):
        self.ds=ds;self.seed=seed;self.epoch=0;self.batch_size=batch_size
        if batch_size not in (4,10):raise ValueError('SupCon microbatch must be 4 or 10')
        self.groups={c:defaultdict(list) for c in range(5)}
        for i,r in enumerate(ds.rows):self.groups[int(r['emotion_id'])][int(r['subject_id'])].append(i)
        if any(len(self.groups[c])<2 for c in range(5)):raise ValueError('Need >=2 speakers for every class')
        perclass=max(sum(len(v) for v in self.groups[c].values()) for c in range(5))
        self.rounds=math.ceil(perclass/(4 if batch_size==4 else 2))
    def __len__(self):return self.rounds*(5 if self.batch_size==4 else 1)
    def __iter__(self):
        rng=np.random.default_rng(stable_seed(self.seed,self.epoch,'eav-speech-supcon',self.batch_size))
        streams={};pos={};counts=Counter()
        for c in range(5):
            for sid,idx in self.groups[c].items():
                key=(c,sid);streams[key]=idx.copy();rng.shuffle(streams[key]);pos[key]=0
        def next_idx(c,sid):
            key=(c,sid)
            if pos[key]>=len(streams[key]):rng.shuffle(streams[key]);pos[key]=0
            z=streams[key][pos[key]];pos[key]+=1;return int(z)
        def pair(c):
            subjects=sorted(self.groups[c])
            weights=np.array([len(self.groups[c][s])/(1+counts[(c,s)]) for s in subjects],float)
            weights/=weights.sum();s1,s2=rng.choice(subjects,size=2,replace=False,p=weights)
            counts[(c,int(s1))]+=1;counts[(c,int(s2))]+=1
            return [next_idx(c,int(s1)),next_idx(c,int(s2))]
        batches=[]
        for _ in range(self.rounds):
            if self.batch_size==10:
                batch=sum((pair(c) for c in range(5)),[])
                rng.shuffle(batch);batches.append(batch)
            else:
                for c in range(5):
                    batch=pair(c)+pair((c+1)%5)
                    rng.shuffle(batch);batches.append(batch)
        rng.shuffle(batches)
        for b in batches:
            assert len(b)==self.batch_size
            yield b


# Original PME4 Architecture A preserved below: GHF (24 Transformer layers),
# DPTP (attention + mean/std), 512d bottleneck, 5-class head, SupCon.

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
        self.emotion_head=nn.Sequential(nn.Linear(512,256),nn.GELU(),nn.Dropout(0.30),nn.Linear(256,5))
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
        if mode:
            if self.stage=='frozen':
                self.wavlm.eval()
            else:
                # The early WavLM modules remain frozen; also disable their
                # dropout. Keep the parent encoder in train mode so HF's
                # gradient checkpointing still applies to trainable layers.
                self.wavlm.feature_extractor.eval()
                self.wavlm.feature_projection.eval()
                for layer in self.wavlm.encoder.layers[:18]:
                    layer.eval()
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




def supcon_loss(projections,targets,tau=.07):
    """Exact supervised contrastive loss used by user's PME4 Architecture A."""
    z=F.normalize(projections.float(),dim=-1,eps=1e-8)
    y=targets.view(-1);logits=(z@z.T)/tau
    diag=torch.eye(len(y),device=y.device,dtype=torch.bool)
    pos=(y[:,None]==y[None,:])&~diag
    if not pos.any(1).all():raise ValueError('SupCon sampler failed: at least one sample has no positive')
    logp=logits.masked_fill(diag,float('-inf'))
    logp=logp-torch.logsumexp(logp,dim=1,keepdim=True)
    loss=-torch.where(pos,logp,torch.zeros_like(logp)).sum(1)/pos.sum(1)
    if not torch.isfinite(loss).all():raise FloatingPointError('Non-finite SupCon')
    return loss.mean()




def quick_scores(frame,prob):
    y=frame.emotion_id.to_numpy(dtype=int);pred=prob.argmax(1)
    from sklearn.metrics import recall_score,f1_score,log_loss
    s=frame.subject_id.to_numpy(dtype=int)
    msa=np.mean([np.mean(pred[s==v]==y[s==v]) for v in sorted(np.unique(s))])
    return {'mean_subject_accuracy':float(msa),
            'uar':float(recall_score(y,pred,average='macro',labels=list(range(5)),zero_division=0)),
            'macro_f1':float(f1_score(y,pred,average='macro',labels=list(range(5)),zero_division=0)),
            'cross_entropy':float(log_loss(y,prob,labels=list(range(5))))}


def history_plot(history,out):
    frame=pd.DataFrame(history);frame.to_csv(out/'train_history.csv',index=False)
    fig,ax=plt.subplots(figsize=(9,5))
    ax.plot(frame.epoch,frame.train_ce,label='Train CE',color='tab:blue')
    ax2=ax.twinx()
    kinds=frame['val_eval_kind'].fillna('full') if 'val_eval_kind' in frame else pd.Series('full',index=frame.index)
    full=kinds=='full'
    quick=kinds=='quick'
    if full.any():
        ax2.plot(frame.loc[full,'epoch'],frame.loc[full,'val_msa']*100,
                 color='tab:orange',marker='o',label='FULL validation MSA (model selection)')
    if quick.any():
        ax2.scatter(frame.loc[quick,'epoch'],frame.loc[quick,'val_msa']*100,
                    color='tab:gray',marker='x',label='Quick subset proxy MSA')
    ax.set(xlabel='Epoch',ylabel='Train CE');ax2.set_ylabel('Val MSA (%)')
    l1,n1=ax.get_legend_handles_labels();l2,n2=ax2.get_legend_handles_labels()
    ax.legend(l1+l2,n1+n2,loc='best',fontsize=8)
    fig.tight_layout();fig.savefig(out/'learning_curves.png',dpi=190);plt.close(fig)


def should_full_validate(epoch, args):
    """Full original-recording evaluation for all selection checkpoints.

    1st epoch, end of frozen stage, 1st finetune epoch, every nth
    epoch within each stage, and final epoch. Proxy epochs never update BEST.
    """
    if args.val_full_every<=1:
        return True
    if epoch in (1,args.frozen_epochs,args.frozen_epochs+1,args.max_epochs):
        return True
    if epoch<args.frozen_epochs:
        return (epoch-1)%args.val_full_every==0
    return (epoch-(args.frozen_epochs+1))%args.val_full_every==0


def smoke():
    torch.manual_seed(42)
    hs=tuple(torch.randn(2,12,64) for _ in range(24))
    mask=torch.tensor([[1]*12,[1]*8+[0]*4],dtype=torch.bool)
    fusion=GroupedHierarchicalFusion(hidden_size=64)
    x,w,g=fusion(hs,mask)
    assert x.shape==(2,12,64) and w.shape==(4,6) and g.shape==(2,4)
    z=DualPathTemporalPooling(dim=64,heads=8)(x,mask)
    assert z.shape==(2,512)
    p=torch.tensor([[1.,0.,0.],[.95,.05,0.],[0.,1.,0.],[0.,.95,.05]],requires_grad=True)
    loss=supcon_loss(p,torch.tensor([0,0,1,1]));loss.backward()
    assert torch.isfinite(loss).item() and p.grad is not None
    for seconds in (4.,5.):
        width=int(seconds*RATE);n=20*RATE
        starts=window_starts(n,width)
        assert len(starts)==int(20/seconds), (seconds,starts)
        assert starts[0]==0 and starts[-1]+width==n
        assert len(set(starts))==len(starts)
        s,_=choose_training_start({'recording_id':'s01_i001'},n,width,0,0,42)
        assert 0<=s<=n-width
    print('SPEECH V2 SMOKE PASS: GHF(24 layers) / DPTP(512) / SupCon / full 20-s views')
    print('Smoke uses random hidden states; pretrained WavLM checkpoint NOT loaded in smoke.')


def parse_args():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--mode',choices=('smoke','audit','prepare','train','test'),default='audit')
    p.add_argument('--data-root',type=Path,default=DEFAULT_DATA)
    p.add_argument('--output-root',type=Path,default=DEFAULT_OUTPUT)
    p.add_argument('--run-name',default='speech_v2',help='Separate output folder below output-root (never speech V1)')
    p.add_argument('--window-seconds',type=float,default=None,help='Default 5.0 (paper-aligned); 4.0 valid ablation')
    p.add_argument('--eval-windows',type=int,default=None,help='0=automatic full-wave coverage; positive overrides count')
    p.add_argument('--train-views-per-recording',type=int,default=None,help='1 default, 2 processes two epochs views per original recording')
    p.add_argument('--noise-probability',type=float,default=None)
    p.add_argument('--head-lr',type=float,default=None)
    p.add_argument('--finetune-head-lr',type=float,default=None)
    p.add_argument('--wavlm-lr',type=float,default=None)
    p.add_argument('--supcon-weight',type=float,default=None)
    p.add_argument('--label-smoothing',type=float,default=None)
    p.add_argument('--frozen-epochs',type=int,default=None)
    p.add_argument('--max-epochs',type=int,default=None)
    p.add_argument('--patience',type=int,default=None)
    p.add_argument('--seed',type=int,default=None)
    p.add_argument('--batch-size',type=int,choices=(4,10),default=None)
    p.add_argument('--grad-accum',type=int,default=None)
    p.add_argument('--checkpoint-revision',default=None)
    p.add_argument('--disable-supcon',action='store_true',default=None)
    p.add_argument('--eval-batch-size',type=int,default=8,
                   help='Inference-only microbatch, safe to change when resuming. 8 fits many A100-40GB setups; try 4 on OOM.')
    p.add_argument('--val-full-every',type=int,default=3,
                   help='Do complete 5-window validation every N epochs; only full checks choose BEST. 1=every epoch.')
    p.add_argument('--val-quick-windows',type=int,default=1,
                   help='Views/recording on non-full validation epochs; 1=center 5-sec view. 0=skip quick validation entirely.')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--resume',action='store_true')
    p.add_argument('--unlock-test',action='store_true')
    a=p.parse_args()
    if a.run_name in ('speech','eeg','eeg_v2','shared','test') or not __import__('re').fullmatch('[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}',a.run_name):
        p.error('Choose a distinct simple --run-name; speech is reserved for V1')
    if a.unlock_test and a.mode!='test':p.error('--unlock-test only in test mode')
    if a.resume and a.mode!='train':p.error('--resume only in train mode')
    if a.eval_batch_size<1 or a.workers<0:p.error('Invalid eval batch size/workers')
    if a.val_full_every<1 or not 0<=a.val_quick_windows<=3:
        p.error('Require val-full-every>=1 and 0<=val-quick-windows<=3')
    return a


def apply_locked_config(args, out):
    path=out/'experiment_config.json'
    if args.mode in ('test',) or args.resume:
        if not path.is_file():raise FileNotFoundError(f'No existing experiment configuration: {path}')
        saved=json.loads(path.read_text())
        if saved.get('format')!=FORMAT:raise RuntimeError('This is not an EAV Speech V2 experiment')
        config=saved['training_settings']
        for key in LOCKED_KEYS:
            explicit=getattr(args,key)
            if explicit is not None and explicit!=config[key]:
                raise ValueError(f'Conflicting --{key.replace("_","-")}: {explicit!r}; saved={config[key]!r}. Use a NEW --run-name')
            setattr(args,key,config[key])
        return saved
    for key,default in DEFAULTS.items():
        if getattr(args,key) is None:setattr(args,key,default)
    if args.mode=='train':
        if path.exists():
            raise RuntimeError(f'Experiment already exists: {out}. Use --resume or new --run-name')
    return None


def validate(args):
    if not 2.0<=args.window_seconds<=8.0:raise ValueError('Require 2<=window-seconds<=8')
    if args.eval_windows<0 or args.eval_windows>15:raise ValueError('eval-windows must be 0..15')
    if args.train_views_per_recording not in (1,2):raise ValueError('train-views-per-recording must be 1 or 2')
    if not 0<=args.noise_probability<=1:raise ValueError('noise probability must be within 0..1')
    if not 1<=args.frozen_epochs<args.max_epochs or args.patience<1:raise ValueError('Invalid epoch schedule')
    if args.batch_size*args.grad_accum!=20:raise ValueError('Effective batch must be 20 (4x5 or 10x2)')
    if not (0<=args.supcon_weight<=.5 and 0<=args.label_smoothing<.3):raise ValueError('Invalid loss weight or smoothing')
    if min(args.head_lr,args.finetune_head_lr,args.wavlm_lr)<=0:raise ValueError('Learning rates must be positive')


def quality_report(df,store,out,width):
    """Descriptive, label-independent QC; NEVER removes recording or windows."""
    qc=[]
    for row in tqdm(df.to_dict('records'),desc='EAV Audio QC',unit='recording'):
        wave=store.get(row)
        starts=window_starts(len(wave),width)
        rms=[]
        for st in starts:
            w=np.asarray(wave[st:st+width],dtype=np.float64)
            rms.append(20*math.log10(max(1e-10,float(np.sqrt(np.mean(w*w))))))
        qc.append({'recording_id':row['recording_id'],'subject_id':row['subject_id'],
                   'split':row['split'],'emotion_id':row['emotion_id'],
                   'duration_seconds':len(wave)/RATE,
                   'n_windows':len(starts),'min_window_rms_dbfs':min(rms),
                   'max_window_rms_dbfs':max(rms),
                   'windows_quiet_below_minus_55_dbfs':sum(v< -55 for v in rms),
                   'peak_amplitude':float(np.max(np.abs(wave)))})
    result=pd.DataFrame(qc);out.mkdir(parents=True,exist_ok=True)
    result.to_csv(out/'audio_quality.csv',index=False)
    summary={'recordings':len(result),'duration_seconds':result.duration_seconds.describe().to_dict(),
             'quiet_windows_total':int(result.windows_quiet_below_minus_55_dbfs.sum()),
             'recordings_with_quiet_windows':int((result.windows_quiet_below_minus_55_dbfs>0).sum()),
             'note':'RMS threshold is a QC diagnostic only, not automatic VAD or exclusion.'}
    json_dump(out/'audio_quality_summary.json',summary)
    return summary


def main(args):
    if args.mode=='smoke':smoke();return
    root=Path(args.output_root).expanduser().resolve()
    out=root/args.run_name;out.mkdir(parents=True,exist_ok=True)
    saved=apply_locked_config(args,out)
    validate(args)
    df,audit=audit_or_load(args.data_root,root/'shared')
    sha=manifest_sha(df)
    if saved is not None and (saved['manifest_sha']!=sha or saved['protocol']!=PROTOCOL):
        raise RuntimeError('Experiment data or original-recording split changed')
    print('[SPEECH V2 AUDIT]',audit['splits'],'classes:',LABELS,flush=True)
    if args.mode=='audit':
        print('AUDIT PASSED; shared immutable split:',root/'shared'/'immutable_original_recording_split.csv');return
    width=window_samples(args)
    store=AudioStore(out/'resampled_audio_16k',cache=not args.no_cache)
    if args.mode=='prepare':
        report=quality_report(df,store,out/'audit',width)
        print('SPEECH V2 PREPARE DONE:',report,flush=True);return
    if not torch.cuda.is_available():raise RuntimeError('Requires NVIDIA CUDA; A100 recommended')
    if not torch.cuda.is_bf16_supported():raise RuntimeError('BF16-capable NVIDIA GPU required')
    import transformers
    from transformers import AutoFeatureExtractor,WavLMModel,WavLMConfig
    seed_all(args.seed)
    device=torch.device('cuda:0')
    torch.backends.cuda.matmul.allow_tf32=True
    torch.backends.cudnn.allow_tf32=True
    print('[GPU]',torch.cuda.get_device_name(0),'Transformers',transformers.__version__,flush=True)
    extractor=AutoFeatureExtractor.from_pretrained(MODEL_NAME,revision=args.checkpoint_revision)
    if extractor.sampling_rate!=RATE:raise RuntimeError(f'Expected WavLM sampling rate 16000, got {extractor.sampling_rate}')
    conf=WavLMConfig.from_pretrained(MODEL_NAME,revision=args.checkpoint_revision)
    if conf.hidden_size!=1024 or conf.num_hidden_layers!=24:raise RuntimeError('Expected pretrained WavLM-Large 1024 dims, 24 layers')
    conf.apply_spec_augment=False;conf.layerdrop=0.0
    wavlm=WavLMModel.from_pretrained(MODEL_NAME,revision=args.checkpoint_revision,config=conf)
    model=ArchitectureA(wavlm).to(device);model.set_stage('frozen')
    try:model.wavlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    except TypeError:model.wavlm.gradient_checkpointing_enable()
    collator=collate(extractor,width)
    train=df[df.split=='train'].sort_values('recording_id').copy()
    val=df[df.split=='val'].sort_values('recording_id').copy()
    ds=TrainDataset(train,store,width,args.seed,args.train_views_per_recording,args.noise_probability)
    sampler=BalancedClassPairSampler(ds,args.seed,args.batch_size)
    settings={k:getattr(args,k) for k in LOCKED_KEYS}
    experimental={'format':FORMAT,'protocol':PROTOCOL,'manifest_sha':sha,
                  'training_settings':settings,'setting_sha':fingerprint(settings),
                  'source_script_sha256':sha_file(__file__),
                  'model':'WavLM-Large + GHF(4 x 6 transformer layers) + DPTP(512) + SupCon',
                  'note':'EAV 42-subject speaker-overlapping 70/15/15 original-recording split; not LOSO.'}
    if args.mode=='train':
        if saved is None:
            json_dump(out/'experiment_config.json',experimental)
        # On --resume, preserve the original source SHA / experiment metadata.
        json_dump(out/'fast_evaluation_policy.json',{
            'note':'Run-time validation scheduling only; original data, architecture and saved locked training hyperparameters unchanged.',
            'val_full_every':args.val_full_every,'val_quick_windows':args.val_quick_windows,
            'eval_batch_size':args.eval_batch_size,'workers':args.workers,
            'full_val_for_best_only':True,
            'test_always_full_recording':True,
            'resume':bool(args.resume),
            'previous_script_sha256':saved.get('source_script_sha256') if saved else None,
            'source_script_sha256':sha_file(__file__)})
    print('[MODEL] train original:',len(train),'train examples:',len(ds),'validation original:',len(val),
          'window seconds:',args.window_seconds,'validation views:', 'full auto' if args.eval_windows==0 else args.eval_windows,
          'samples/epoch:',len(sampler)*args.batch_size,flush=True)

    def infer(frame,which,epoch=None,override_windows=None):
        if which=='test' and not(args.mode=='test' and args.unlock_test):raise PermissionError('Test partition locked')
        model.eval()
        eval_ds=EvalDataset(frame,store,width,args.eval_windows if override_windows is None else override_windows)
        loader=DataLoader(eval_ds,batch_size=args.eval_batch_size,num_workers=args.workers,
                          pin_memory=True,collate_fn=collator,persistent_workers=False)
        sums={};count=Counter()
        with torch.inference_mode():
            for b in tqdm(loader,desc=f'[SPEECH V2 EVAL] {which}',unit='batch'):
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(b['input_values'].to(device,non_blocking=True),
                                 b['attention_mask'].to(device,non_blocking=True))
                logits=output['logits'].float().cpu().numpy()
                if not np.isfinite(logits).all():raise FloatingPointError('Nonfinite WavLM logits')
                for i,rid in enumerate(b['recording_id']):
                    if rid not in sums:sums[rid]=np.zeros(5,dtype=np.float64)
                    sums[rid]+=logits[i];count[rid]+=1
        frame=frame.sort_values('recording_id').reset_index(drop=True)
        if set(frame.recording_id)!=set(sums):raise RuntimeError('Prediction coverage mismatch')
        logits=np.stack([sums[r]/count[r] for r in frame.recording_id])
        probs=softmax(logits,axis=-1)
        pred=frame[['recording_id','subject_id','trial_id','emotion_id','emotion']].copy()
        pred['n_windows']=[count[r] for r in frame.recording_id]
        for c,slug in enumerate(SLUGS):pred[f'p_{slug}']=probs[:,c]
        return pred,probs

    def save_ckpt(path,payload):
        temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
        torch.save(payload,temp);os.replace(temp,path)

    def checkpoint_payload(epoch,stage,history,best_ep,best_rank,stale,opt=None,sched=None):
        d={'format':FORMAT,'protocol':PROTOCOL,'epoch':epoch,'stage':stage,
           'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
           'manifest_sha':sha,'settings_sha':fingerprint(settings),
           'checkpoint_revision':args.checkpoint_revision,'rng':rng_capture(),
           'history':history,'best_epoch':best_ep,'best_rank':best_rank,'stale':stale}
        if opt is not None:d.update({'optimizer':opt.state_dict(),'scheduler':sched.state_dict()})
        return d

    def load_ckpt(file,restore_rng=False):
        ck=torch.load(file,map_location='cpu',weights_only=False) # ONLY own trusted experiment checkpoints
        if ck.get('format')!=FORMAT or ck.get('manifest_sha')!=sha or ck.get('settings_sha')!=fingerprint(settings):
            raise RuntimeError('Speech V2 model/data/config mismatch. Do not resume V1 or use altered training settings.')
        model.load_state_dict(ck['model'],strict=True);model.set_stage(ck['stage'])
        if restore_rng:rng_restore(ck['rng'])
        return ck

    if args.mode=='test':
        if not args.unlock_test:raise PermissionError('Test requires --mode test --unlock-test')
        info_path=out/'training_complete.json'
        if not info_path.exists():raise RuntimeError('Training is incomplete; test remains locked')
        test_dir=out/'test';test_dir.mkdir(exist_ok=True)
        started=test_dir/'final_test_started.lock'
        if started.exists() or (test_dir/'final_test_completed.json').exists():
            raise RuntimeError('Final test already started/completed; do not repeat test-set evaluation')
        info=json.loads(info_path.read_text())
        if info['manifest_sha']!=sha or info['best_checkpoint_sha']!=sha_file(out/'best.pth'):
            raise RuntimeError('Selected checkpoint or split changed after model selection')
        ck=load_ckpt(out/'best.pth')
        with open(started,'x') as lock:
            lock.write('Final test started at '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'\n')
        test=df[df.split=='test'].copy()
        prediction,_=infer(test,'test',ck['epoch'])
        met=report_predictions(prediction,test_dir,'test',epoch=ck['epoch'])
        json_dump(test_dir/'final_test_completed.json',{'test_evaluated_once':True,
                  'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  'manifest_sha':sha,'selected_epoch':ck['epoch'],'metrics':met,'protocol':PROTOCOL,
                  'checkpoint_sha':sha_file(out/'best.pth')})
        print(f'FINAL EAV SPEECH V2 TEST MSA={met["mean_subject_accuracy"]:.2%}, macro-F1={met["macro_f1"]:.2%}')
        return

    if args.mode!='train':raise ValueError(args.mode)
    if (out/'training_complete.json').exists():raise RuntimeError('This run already finished; choose a new --run-name')
    start=1;best_epoch=0;best_rank=None;stale=0;history=[];old=None
    if args.resume:
        old=load_ckpt(out/'last.pth',restore_rng=True)
        start=old['epoch']+1;best_epoch=old['best_epoch']
        best_rank=tuple(old['best_rank']) if old['best_rank'] is not None else None
        stale=old['stale'];history=old['history']
    stage_prev=None;opt=None;sched=None
    updates_per_epoch=math.ceil(len(sampler)/args.grad_accum)

    def make_optimizer(stage):
        groups=defaultdict(list)
        for name,param in model.named_parameters():
            if not param.requires_grad:continue
            if name.startswith('wavlm.encoder.layers.'):
                idx=int(name.split('.')[3]);lr=args.wavlm_lr*(.8**(23-idx))
            else:lr=args.head_lr if stage=='frozen' else args.finetune_head_lr
            decay=.01 if param.ndim>=2 and not name.endswith('bias') and 'layer_logits' not in name else 0.
            groups[(lr,decay)].append(param)
        return torch.optim.AdamW([{'params':v,'lr':lr,'weight_decay':decay} for (lr,decay),v in groups.items()],eps=1e-8)

    def make_sched(optimizer,stage):
        span=args.frozen_epochs if stage=='frozen' else args.max_epochs-args.frozen_epochs
        total=max(1,span*updates_per_epoch);warm=max(1,int(total*.1))
        def schedule(x):
            if x<warm:return max(.05,(x+1)/warm)
            progress=min(1,(x-warm)/max(1,total-warm))
            return .05+.95*(1+math.cos(math.pi*progress))/2
        return torch.optim.lr_scheduler.LambdaLR(optimizer,schedule)

    for epoch in range(start,args.max_epochs+1):
        stage='frozen' if epoch<=args.frozen_epochs else 'finetune'
        if stage!=stage_prev:
            model.set_stage(stage);opt=make_optimizer(stage);sched=make_sched(opt,stage)
            if old is not None and old['stage']==stage:
                opt.load_state_dict(old['optimizer']);sched.load_state_dict(old['scheduler'])
                for state in opt.state.values():
                    for k,v in state.items():
                        if torch.is_tensor(v):state[k]=v.to(device)
            old=None;stage_prev=stage
            print('[SPEECH V2] stage:',stage,'trainable:',sum(p.numel() for p in model.parameters() if p.requires_grad),flush=True)
        ds.epoch=epoch-1;sampler.epoch=epoch-1
        loader=DataLoader(ds,batch_sampler=sampler,num_workers=args.workers,
                          pin_memory=True,collate_fn=collator)
        model.train();opt.zero_grad(set_to_none=True)
        seen=correct=0;total=0.;ce_total=0.;sup_total=0.
        sup_weight=0. if args.disable_supcon or stage=='frozen' else min(args.supcon_weight,args.supcon_weight*(epoch-args.frozen_epochs)/3)
        for i,b in enumerate(tqdm(loader,desc=f'[SPEECH V2 TRAIN] {epoch}/{args.max_epochs}',unit='batch')):
            y=b['y'].to(device,non_blocking=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                output=model(b['input_values'].to(device,non_blocking=True),
                    b['attention_mask'].to(device,non_blocking=True),need_projection=bool(sup_weight))
            logits=output['logits'].float()
            ce=F.cross_entropy(logits,y,label_smoothing=args.label_smoothing)
            sup=supcon_loss(output['projection'],y) if sup_weight else ce.new_zeros(())
            loss=ce+sup_weight*sup
            if not torch.isfinite(loss).item():raise FloatingPointError('Nonfinite training loss')
            effective=min(args.grad_accum,len(loader)-(i//args.grad_accum)*args.grad_accum)
            (loss/effective).backward()
            if (i+1)%args.grad_accum==0 or i+1==len(loader):
                torch.nn.utils.clip_grad_norm_([q for q in model.parameters() if q.requires_grad],1.)
                opt.step();sched.step();opt.zero_grad(set_to_none=True)
            seen+=len(y);correct+=int((logits.argmax(1)==y).sum())
            total+=float(loss.detach())*len(y);ce_total+=float(ce.detach())*len(y);sup_total+=float(sup.detach())*len(y)
        full_eval=should_full_validate(epoch,args)
        # A quick validation window is monitoring-only and never influences
        # checkpoint selection, LR, early stop or final metrics.
        if full_eval:
            preds,probs=infer(val,'val',epoch,override_windows=args.eval_windows)
            val_scores=quick_scores(val,probs)
            kind='full'
        elif args.val_quick_windows:
            preds,probs=infer(val,'val_quick',epoch,override_windows=args.val_quick_windows)
            val_scores=quick_scores(val,probs)
            kind='quick'
        else:
            preds=None;val_scores=None;kind='skipped'
        row={'epoch':epoch,'stage':stage,'train_loss':total/seen,'train_ce':ce_total/seen,
             'train_supcon':sup_total/seen,'train_accuracy':correct/seen,
             'val_eval_kind':kind,
             'val_msa':val_scores['mean_subject_accuracy'] if val_scores else float('nan'),
             'val_uar':val_scores['uar'] if val_scores else float('nan'),
             'val_macro_f1':val_scores['macro_f1'] if val_scores else float('nan')}
        history.append(row);history_plot(history,out)
        improved=False
        if full_eval:
            rank=(val_scores['mean_subject_accuracy'],val_scores['uar'],val_scores['macro_f1'],
                  -val_scores['cross_entropy'],-epoch)
            if best_rank is None or rank>best_rank:
                best_rank=rank;best_epoch=epoch;stale=0;improved=True
                save_ckpt(out/'best.pth',checkpoint_payload(epoch,stage,history,best_epoch,best_rank,stale))
                preds.to_csv(out/'best_validation_predictions.csv',index=False)
        if stage=='finetune' and not improved:
            stale+=1  # count stale training epochs; evaluate stopping ONLY at a full check
        save_ckpt(out/'last.pth',checkpoint_payload(epoch,stage,history,best_epoch,best_rank,stale,opt,sched))
        metric=f'{row["val_msa"]:.2%}' if val_scores else 'not run'
        print(f'[SPEECH V2 FAST EPOCH {epoch:03d}] train={row["train_accuracy"]:.2%} '
              f'VAL {kind} MSA={metric} best_FULL_epoch={best_epoch} stale_epochs={stale}',flush=True)
        if full_eval and stage=='finetune' and stale>=args.patience:
            print('[EARLY STOP] No FULL validation MSA improvement for >=',
                  args.patience,'finetune epochs')
            break
    ck=load_ckpt(out/'best.pth')
    best_preds,_=infer(val,'val',ck['epoch'])
    metrics=report_predictions(best_preds,out/'val','val',epoch=ck['epoch'])
    with torch.no_grad():weights=F.softmax(model.ghf.layer_logits.float(),dim=-1).cpu().numpy()
    pd.DataFrame([{'group':gi+1,'wavlm_layer':gi*6+j+1,'weight':float(weights[gi,j])}
        for gi in range(4) for j in range(6)]).to_csv(out/'ghf_layer_weights.csv',index=False)
    json_dump(out/'training_complete.json',{'best_epoch':ck['epoch'],
              'best_val_metrics':metrics,'manifest_sha':sha,
              'settings_sha':fingerprint(settings),'best_checkpoint_sha':sha_file(out/'best.pth'),
              'test_locked':True,'subject_dependent':True})
    print('SPEECH V2 TRAIN COMPLETE; final test remains LOCKED.',flush=True)


if __name__=='__main__':
    try:main(parse_args())
    except KeyboardInterrupt:
        print('Interrupted. To resume: --mode train --resume with SAME --run-name',file=sys.stderr)
        sys.exit(130)
    except Exception:
        traceback.print_exc();sys.exit(1)
