"""Checkpoint-compatible 256-dimensional attention fusion."""


import torch
from torch import nn
import timm


class SomaTokenEncoder(nn.Module):
    def __init__(self, feature_cols, hidden_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.feature_cols = feature_cols
        self.num_tokens = len(feature_cols)

        self.protein_emb = nn.Embedding(self.num_tokens, hidden_dim)

        self.expr_proj = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        enc_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)

        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x):
        # x (B, F)
        B, F = x.shape
        idx = torch.arange(F, device=x.device).unsqueeze(0).expand(B, -1)
        prot = self.protein_emb(idx)                # (B, F, D)
        expr = self.expr_proj(x.unsqueeze(-1))      # (B, F, D)

        tokens = prot + expr                         # (B, F, D)
        cls = self.cls.expand(B, 1, -1)
        tokens = torch.cat([cls, tokens], dim=1)     # (B, F+1, D)

        h = self.encoder(tokens)[:, 0]               # (B, D) CLS
        return h


class TransformerModule(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, nhead=4, nlayers=3):
        super().__init__()
        self.proj = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x):
        # (B, F) → treat as seq length=1
        x = self.proj(x).unsqueeze(1)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.encoder(x)[:, 0]


class ECGTransformer(nn.Module):
    def __init__(self, input_dim=12, hidden_dim=128, nhead=4, nlayers=3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, nlayers)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))

    def forward(self, x):
        # x: (B, T, 12)
        x = self.embedding(x)
        B = x.size(0)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        return self.encoder(x)[:, 0]


class FusionModel(nn.Module):
    def __init__(self, use_fundus, use_snp, use_soma, use_ecg,
                 snp_dim=0, feature_cols=None, init_ckpts=None, use_pretrained_ckpt=True):
        super().__init__()
        from pipelines.checkpoint_io import load_matching
        self.use_fundus, self.use_snp = use_fundus, use_snp
        self.use_soma, self.use_ecg = use_soma, use_ecg
        self.ln = nn.LayerNorm(256)
        self.initialization = {}
        if use_fundus:
            self.fundus_enc = timm.create_model("vit_small_patch16_224", pretrained=False)
            self.fundus_enc.reset_classifier(0)
            self.fundus_proj = nn.Linear(self.fundus_enc.num_features, 256)
            self.fundus_proj.requires_grad_(False)
        if use_snp:
            self.snp_enc = TransformerModule(snp_dim, hidden_dim=256, nlayers=4)
        if use_soma:
            self.soma_enc = SomaTokenEncoder(feature_cols, hidden_dim=256, nlayers=8)
        if use_ecg:
            self.ecg_enc = ECGTransformer(input_dim=12, hidden_dim=256, nlayers=3)
        self.ln.requires_grad_(False)
        self.attn_net = nn.Sequential(nn.Linear(256, 64), nn.Tanh(), nn.Linear(64, 1))
        self.mlp = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 1))
        if use_pretrained_ckpt:
            for name in self.modalities:
                if not init_ckpts or name not in init_ckpts:
                    raise ValueError("Missing initialization checkpoint for " + name)
                self.initialization[name] = load_matching(
                    getattr(self, name + "_enc"), init_ckpts[name], encoder=True,
                    aliases={"proj.": "embedding."} if name == "ecg" else None)

    @property
    def modalities(self):
        return [name for name in ("fundus", "snp", "soma", "ecg") if getattr(self, "use_" + name)]

    def extract_features(self, fundus, snp, soma, ecg):
        inputs = {"fundus": fundus, "snp": snp, "soma": soma, "ecg": ecg}
        single = {}
        for name in self.modalities:
            value = getattr(self, name + "_enc")(inputs[name])
            if name == "fundus":
                value = self.fundus_proj(value)
            single[name] = self.ln(value)
        stacked = torch.stack(list(single.values()), dim=1)
        weights = torch.softmax(self.attn_net(stacked), dim=1)
        return single, (stacked * weights).sum(1)

    def forward(self, fundus, snp, soma, ecg):
        _, fused = self.extract_features(fundus, snp, soma, ecg)
        return self.mlp(fused).squeeze(-1)

    def parameter_groups(self, encoder_lr=1e-6, head_lr=1e-3):
        groups = [{"params": getattr(self, name + "_enc").parameters(),
                   "lr": encoder_lr, "name": name} for name in self.modalities]
        groups.extend({"params": getattr(self, name).parameters(), "lr": head_lr, "name": name}
                      for name in ("attn_net", "mlp"))
        return groups
