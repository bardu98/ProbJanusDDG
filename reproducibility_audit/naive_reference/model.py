"""Full-sequence one-hot FFNN with masked pooling and mutation-site readout."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence

ALPHABET = 'ACDEFGHIKLMNPQRSTVWY'
EPS = 1e-3


def encode(frame):
    """Losslessly encode complete sequences as indices; one-hot is made in forward.

    Mutation centers come from sequence differences, not possibly remapped numbering.
    No truncation, fitted vocabulary, pretrained features, or target-derived features.
    """
    lookup = {aa:i for i,aa in enumerate(ALPHABET)}
    wild, mutant, centers = [], [], []
    for wt, mt in zip(frame.wt_seq,frame.mut_seq):
        if len(wt)!=len(mt): raise ValueError('Only substitutions supported')
        changed=[j for j,(a,b) in enumerate(zip(wt,mt)) if a!=b]
        if len(changed)!=1: raise ValueError('Exactly one substitution required')
        wild.append(torch.tensor([lookup[x] for x in wt],dtype=torch.long))
        mutant.append(torch.tensor([lookup[x] for x in mt],dtype=torch.long))
        centers.append(changed[0])
    return wild,mutant,np.array(centers,dtype=np.int64)


def batch(wild,mutant,centers,indices,device):
    ids=[int(i) for i in indices]
    lengths=torch.tensor([len(wild[i]) for i in ids],device=device)
    w=pad_sequence([wild[i] for i in ids],batch_first=True,padding_value=20).to(device)
    m=pad_sequence([mutant[i] for i in ids],batch_first=True,padding_value=20).to(device)
    c=torch.as_tensor(np.asarray(centers)[ids],device=device)
    return w,m,lengths,c


class NaiveGaussianFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.residue=nn.Sequential(nn.Linear(42,64),nn.ReLU(),nn.Linear(64,64),nn.ReLU())
        self.readout=nn.Sequential(nn.Linear(192,64),nn.ReLU(),nn.Linear(64,2))

    def raw(self,wild,mutant,lengths,centers):
        positions=torch.arange(wild.shape[1],device=wild.device)[None,:]
        mask=positions<lengths[:,None]
        # Padding gets its own temporary channel, discarded before the FFNN.
        w=F.one_hot(wild,num_classes=21)[...,:20].float()
        m=F.one_hot(mutant,num_classes=21)[...,:20].float()
        relative=(positions-centers[:,None]).float()/lengths[:,None]
        x=torch.cat([w,w-m,relative[...,None],relative.abs()[...,None]],dim=-1)
        h=self.residue(x)
        mean=(h*mask[...,None]).sum(1)/lengths[:,None]
        maximum=h.masked_fill(~mask[...,None],-torch.inf).amax(1)
        site=h[torch.arange(len(h),device=h.device),centers]
        return self.readout(torch.cat([mean,maximum,site],dim=-1))

    def forward(self,wild,mutant,lengths,centers,directional=False):
        ab=self.raw(wild,mutant,lengths,centers)
        if directional: return ab[:,0],F.softplus(ab[:,1])+EPS
        ba=self.raw(mutant,wild,lengths,centers)
        return (ab[:,0]-ba[:,0])/2,F.softplus((ab[:,1]+ba[:,1])/2)+EPS


@torch.no_grad()
def predict(model,wild,mutant,centers,bs=16):
    model.eval(); device=next(model.parameters()).device
    mu=np.zeros(len(wild),dtype=np.float32); sigma=np.zeros_like(mu)
    order=np.argsort([-len(w) for w in wild],kind='stable')
    for start in range(0,len(order),bs):
        ids=order[start:start+bs]
        m,s=model(*batch(wild,mutant,centers,ids,device))
        mu[ids]=m.cpu().numpy(); sigma[ids]=s.cpu().numpy()
    if not np.isfinite([mu,sigma]).all() or not (sigma>0).all():
        raise ValueError('Invalid predictions')
    return mu,sigma
