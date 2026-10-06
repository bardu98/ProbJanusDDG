"""
Architettura: tronco cross-attention su embedding ESM-2 congelati, con due teste di lettura.

ADDESTRAMENTO E INFERENZA SONO DIVERSI, come nel JanusDDG originale
(`data/janusddg/train.ipynb` cella 7 contro `data/janusddg/model.py`):

    addestramento   uscita = il solo passaggio DIRETTO, m_AB e softplus(s_AB)
                    sul dataset RADDOPPIATO con i versi invertiti ed etichette negate

    inferenza       mu    = ( m_AB - m_BA ) / 2                   antisimmetrico esatto
                    sigma = softplus( (s_AB + s_BA) / 2 ) + EPS   simmetrico esatto

PERCHE'. Scomponendo l'uscita direzionale in parte antisimmetrica e simmetrica,
f = f_A + f_S, antisimmetrizzare DENTRO l'addestramento fa vedere alla perdita solo f_A:
f_S resta una direzione nulla, libera di assorbire capacita' e derivare senza vincolo.
Addestrando invece il modello direzionale sui due versi, con perdita quadratica

    (y - f(A,B))^2 + (-y - f(B,A))^2  =  2 (y - f_A)^2  +  2 f_S^2

cioe' la stessa perdita PIU' una penalizzazione che azzera la parte simmetrica. Per sigma
vale lo speculare: conta la parte simmetrica di s, e il verso invertito azzera quella
antisimmetrica.

Corollario: con l'antisimmetrizzazione in addestramento il dataset invertito sarebbe una
no-op esatta (la perdita su (B,A,-y) e' algebricamente identica a quella su (A,B,y)), quindi
raddoppierebbe il costo senza aggiungere informazione. Il verso invertito E' il meccanismo.

Le uscite grezze delle due direzioni sono combinate PRIMA della softplus: combinarle dopo non
darebbe sigma(A,B) = sigma(B,A) come identita' esatta.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DIM, EPS


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_len=3700):
        super().__init__()
        pe = torch.zeros(max_len, embedding_dim)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, embedding_dim, 2).float()
                        * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


def masked_pooling(h, padding_mask):
    """media e massimo globali sui soli residui validi."""
    pm = padding_mask.float().unsqueeze(-1)
    valid = (1 - padding_mask.float()).sum(1).clamp(min=1)
    gap = (h * (1 - pm)).sum(1) / valid.unsqueeze(-1)
    gmp = (h * (1 - pm) + pm * (-1e10)).max(1).values
    return gap, gmp


class Trunk(nn.Module):
    """Un passaggio direzionale: legge (delta, riferimento) e restituisce due scalari grezzi."""

    def __init__(self, input_dim=DIM, num_heads=8, dropout_rate=0.0,
                 f_activation=None, kernel_size=20, out_channels=128,
                 correct_conv_mask=False):
        super().__init__()
        act = f_activation if f_activation is not None else nn.ReLU()
        self.conv1d = nn.Conv1d(input_dim, out_channels, kernel_size, padding=0)
        self.conv1d_wild = nn.Conv1d(input_dim, out_channels, kernel_size, padding=0)
        self.kernel_size = kernel_size
        self.correct_conv_mask = correct_conv_mask
        self.norm1 = nn.LayerNorm(out_channels)
        self.norm2 = nn.LayerNorm(out_channels)
        self.positional_encoding = SinusoidalPositionalEncoding(out_channels, 3700)
        self.multihead_attention = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout_rate, batch_first=True)
        self.inverse_attention = nn.MultiheadAttention(
            out_channels, num_heads, dropout=dropout_rate, batch_first=True)
        d2 = out_channels * 2
        self.norm3 = nn.LayerNorm(d2)
        self.pw_ffnn = nn.Sequential(nn.Linear(d2, 512), act, nn.Linear(512, d2))

        self.Linear_ddg = nn.Linear(d2 * 2, 1)      # readout di mu
        self.Linear_sig = nn.Linear(d2 * 2, 1)      # readout di sigma
        # INIT CASUALE, per decisione di Guido: Linear_sig resta con l'inizializzazione di serie
        # di nn.Linear (kaiming_uniform sul peso, uniform(+-1/sqrt(fan_in)) sul bias), la stessa
        # di ogni altro Linear della rete. L'azzeramento storico e' stato tolto: l'ablazione
        # appaiata in analysis/ablation_initsigma lo dava perdente su S461L (P=99,8%).
        pass

    @staticmethod
    def padding_mask(length, seq_len):
        return torch.arange(seq_len, device=length.device).unsqueeze(0) >= length.unsqueeze(1)

    def pooled(self, delta, x_ref, length):
        c_delta = self.positional_encoding(self.conv1d(delta.transpose(1, 2)).transpose(1, 2))
        c_ref = self.positional_encoding(self.conv1d_wild(x_ref.transpose(1, 2)).transpose(1, 2))
        # Una convoluzione valid con kernel K produce length-K+1 finestre reali.
        # Il comportamento storico resta disponibile con correct_conv_mask=False.
        mask_length = ((length - self.kernel_size + 1).clamp(min=0)
                       if self.correct_conv_mask else length)
        mask = self.padding_mask(mask_length, c_ref.size(1))

        direct, _ = self.multihead_attention(c_ref, c_delta, c_delta, key_padding_mask=mask)
        direct = self.norm1(direct + c_delta)
        inverse, _ = self.inverse_attention(c_delta, c_ref, c_ref, key_padding_mask=mask)
        inverse = self.norm2(inverse + c_ref)

        attn = torch.cat([direct, inverse], dim=-1)
        h = self.norm3(attn + self.pw_ffnn(attn))
        gap, gmp = masked_pooling(h, mask)
        return torch.cat([gap, gmp], dim=-1)

    def forward(self, delta, x_ref, length):
        h = self.pooled(delta, x_ref, length)
        return self.Linear_ddg(h).squeeze(-1), self.Linear_sig(h).squeeze(-1)


class ProbabilisticJanus(nn.Module):
    """Il modello completo. Ogni coppia passa nel tronco due volte, diretta e rovesciata."""

    def __init__(self, **kw):
        super().__init__()
        self.base_ddg = Trunk(**kw)

    def forward(self, x_wild, x_mut, length, train=False):
        """train=True: solo passaggio diretto. train=False: antisimmetrizzato (cio' che si consegna).

        La validazione, le out-of-fold e il test usano SEMPRE train=False: il numero riportato
        deve essere quello che il modello consegna, non quello su cui si ottimizza.
        """
        m_ab, s_ab = self.base_ddg(x_wild - x_mut, x_wild, length)
        if train:
            return m_ab, F.softplus(s_ab) + EPS
        m_ba, s_ba = self.base_ddg(x_mut - x_wild, x_mut, length)
        mu = (m_ab - m_ba) / 2
        sigma = F.softplus((s_ab + s_ba) / 2) + EPS
        return mu, sigma


def build(device="cuda", correct_conv_mask=False):
    return ProbabilisticJanus(input_dim=DIM, num_heads=8, dropout_rate=0.0,
                              f_activation=nn.ReLU(), kernel_size=20,
                              correct_conv_mask=correct_conv_mask).to(device)
