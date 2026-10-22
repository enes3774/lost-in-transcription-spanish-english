"""Plain-PyTorch port of the NeMo Conformer-CTC (large, rel_pos, striding x4) used for the enes
code-switch model, so the Miami-fine-tuned weights run in the competition runtime without NeMo.
Needs only torch + sentencepiece. Weights: the NeMo model_weights.ckpt state dict, loaded 1:1.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Featurizer(nn.Module):
    """AudioToMelSpectrogramPreprocessor: preemph 0.97, STFT 512/400/160 (hann, center, reflect),
    power 2, mel fb from the checkpoint, log(x + 2^-24), per-feature normalisation over valid frames."""

    def __init__(self, n_fft=512, win=400, hop=160, n_mels=80):
        super().__init__()
        self.n_fft, self.win, self.hop = n_fft, win, hop
        self.register_buffer("window", torch.hann_window(win, periodic=False))
        self.register_buffer("fb", torch.zeros(1, n_mels, n_fft // 2 + 1))

    def forward(self, x, lengths):
        seq_len = lengths // self.hop + 1
        x = torch.cat((x[:, :1], x[:, 1:] - 0.97 * x[:, :-1]), dim=1)
        spec = torch.stft(x, self.n_fft, hop_length=self.hop, win_length=self.win, window=self.window,
                          center=True, pad_mode="reflect", return_complex=True)
        spec = spec.real ** 2 + spec.imag ** 2
        x = torch.matmul(self.fb.to(spec.dtype), spec)
        x = torch.log(x + 2 ** -24)
        B, D, T = x.shape
        mask = torch.arange(T, device=x.device)[None, :] < seq_len[:, None]  # (B, T) valid
        m = mask[:, None, :].to(x.dtype)
        n = seq_len.to(x.dtype)[:, None, None]
        mean = (x * m).sum(2, keepdim=True) / n
        var = (((x - mean) * m) ** 2).sum(2, keepdim=True) / (n - 1)
        x = (x - mean) / (var.sqrt() + 1e-5)
        x = x.masked_fill(~mask[:, None, :], 0.0)
        return x, seq_len


class RelPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.extend(max_len)

    def extend(self, length):
        pos = torch.arange(length - 1, -length, -1.0).unsqueeze(1)
        div = torch.exp(torch.arange(0, self.d_model, 2.0) * -(math.log(10000.0) / self.d_model))
        pe = torch.zeros(2 * length - 1, self.d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)
        self.max_len = length

    def forward(self, T):
        if T > self.max_len:
            self.extend(T)
        c = self.pe.size(1) // 2
        return self.pe[:, c - T + 1: c + T]


class RelPosMHA(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.h, self.d_k = n_heads, d_model // n_heads
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = nn.Parameter(torch.zeros(n_heads, self.d_k))
        self.pos_bias_v = nn.Parameter(torch.zeros(n_heads, self.d_k))

    @staticmethod
    def rel_shift(x):
        b, h, qlen, pos_len = x.size()
        x = F.pad(x, (1, 0))
        x = x.view(b, h, -1, qlen)
        return x[:, :, 1:].view(b, h, qlen, pos_len)

    def forward(self, x, pos_emb, att_mask):
        B, T, _ = x.shape
        q = self.linear_q(x).view(B, T, self.h, self.d_k)
        k = self.linear_k(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(x).view(B, T, self.h, self.d_k).transpose(1, 2)
        p = self.linear_pos(pos_emb).view(1, -1, self.h, self.d_k).transpose(1, 2)
        q_u = (q + self.pos_bias_u).transpose(1, 2)
        q_v = (q + self.pos_bias_v).transpose(1, 2)
        ac = torch.matmul(q_u, k.transpose(-2, -1))
        bd = self.rel_shift(torch.matmul(q_v, p.transpose(-2, -1)))[:, :, :, :T]
        scores = (ac + bd) / math.sqrt(self.d_k)
        scores = scores.masked_fill(att_mask[:, None], -10000.0)
        attn = torch.softmax(scores, dim=-1).masked_fill(att_mask[:, None], 0.0)
        out = torch.matmul(attn, v).transpose(1, 2).reshape(B, T, -1)
        return self.linear_out(out)


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self.linear2(F.silu(self.linear1(x)))


class ConvModule(nn.Module):
    def __init__(self, d_model, kernel):
        super().__init__()
        self.pointwise_conv1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.depthwise_conv = nn.Conv1d(d_model, d_model, kernel, padding=(kernel - 1) // 2, groups=d_model)
        self.batch_norm = nn.BatchNorm1d(d_model)
        self.pointwise_conv2 = nn.Conv1d(d_model, d_model, 1)

    def forward(self, x, pad_mask):
        x = x.transpose(1, 2)
        x = F.glu(self.pointwise_conv1(x), dim=1)
        x = x.masked_fill(pad_mask[:, None, :], 0.0)
        x = self.depthwise_conv(x)
        x = F.silu(self.batch_norm(x))
        x = self.pointwise_conv2(x)
        return x.transpose(1, 2)


class ConformerLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_heads, kernel):
        super().__init__()
        self.norm_feed_forward1 = nn.LayerNorm(d_model)
        self.feed_forward1 = FeedForward(d_model, d_ff)
        self.norm_self_att = nn.LayerNorm(d_model)
        self.self_attn = RelPosMHA(d_model, n_heads)
        self.norm_conv = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, kernel)
        self.norm_feed_forward2 = nn.LayerNorm(d_model)
        self.feed_forward2 = FeedForward(d_model, d_ff)
        self.norm_out = nn.LayerNorm(d_model)

    def forward(self, x, pos_emb, att_mask, pad_mask):
        r = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        r = r + self.self_attn(self.norm_self_att(r), pos_emb, att_mask)
        r = r + self.conv(self.norm_conv(r), pad_mask)
        r = r + 0.5 * self.feed_forward2(self.norm_feed_forward2(r))
        return self.norm_out(r)


class ConvSubsampling(nn.Module):
    """striding x4: two Conv2d(k3, s2, p1)+ReLU, then Linear(512*20 -> 512)."""

    def __init__(self, feat_in, d_model, channels=512):
        super().__init__()
        self.conv = nn.Sequential(nn.Conv2d(1, channels, 3, 2, 1), nn.ReLU(), nn.Conv2d(channels, channels, 3, 2, 1), nn.ReLU())
        f = feat_in
        for _ in range(2):
            f = (f + 2 - 3) // 2 + 1
        self.out = nn.Linear(channels * f, d_model)

    def forward(self, x, lengths):  # x (B, T, F)
        x = self.conv(x.unsqueeze(1))
        B, C, T, Fd = x.shape
        x = self.out(x.transpose(1, 2).reshape(B, T, C * Fd))
        for _ in range(2):
            lengths = torch.div(lengths + 2 - 3, 2, rounding_mode="floor") + 1
        return x, lengths


class ConformerCTC(nn.Module):
    def __init__(self, n_layers=18, d_model=512, n_heads=8, ff=4, kernel=31, feat_in=80, n_classes=1024):
        super().__init__()
        self.preprocessor = nn.Module()
        self.preprocessor.featurizer = Featurizer(n_mels=feat_in)
        self.encoder = nn.Module()
        self.encoder.pre_encode = ConvSubsampling(feat_in, d_model)
        self.encoder.layers = nn.ModuleList([ConformerLayer(d_model, d_model * ff, n_heads, kernel) for _ in range(n_layers)])
        self.pos_enc = RelPositionalEncoding(d_model)
        self.xscale = math.sqrt(d_model)
        self.decoder = nn.Module()
        self.decoder.decoder_layers = nn.Sequential(nn.Conv1d(d_model, n_classes + 1, 1))
        self.blank = n_classes

    @torch.no_grad()
    def forward(self, wav, lengths):
        feats, flen = self.preprocessor.featurizer(wav, lengths)
        x, elen = self.encoder.pre_encode(feats.transpose(1, 2), flen)
        x = x * self.xscale
        T = x.size(1)
        pos_emb = self.pos_enc(T).to(x.dtype)
        valid = torch.arange(T, device=x.device)[None, :] < elen[:, None]
        att = valid[:, None, :] & valid[:, :, None]
        att_mask, pad_mask = ~att, ~valid
        for layer in self.encoder.layers:
            x = layer(x, pos_emb, att_mask, pad_mask)
        logits = self.decoder.decoder_layers(x.transpose(1, 2)).transpose(1, 2)  # (B, T, C+1)
        return torch.log_softmax(logits, dim=-1), elen

    def greedy_ids(self, log_probs, elen):
        out = []
        for lp, n in zip(log_probs, elen):
            ids = lp[:n].argmax(-1).tolist()
            seq, prev = [], None
            for i in ids:
                if i != prev and i != self.blank:
                    seq.append(i)
                prev = i
            out.append(seq)
        return out


def load_conformer(state_dict_path, device="cuda", dtype=torch.float32):
    m = ConformerCTC()
    sd = torch.load(state_dict_path, map_location="cpu")
    sd = {k: v for k, v in sd.items() if not k.endswith("num_batches_tracked")}
    missing, unexpected = m.load_state_dict(sd, strict=False)
    missing = [k for k in missing if "pos_enc" not in k]
    assert not missing and not unexpected, (missing[:5], unexpected[:5])
    return m.to(device=device, dtype=dtype).eval()
