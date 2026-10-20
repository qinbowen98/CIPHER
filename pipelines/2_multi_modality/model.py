# model.py
import torch
import torch.nn as nn
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


########################################
# Utility: Flexible checkpoint loader
########################################
def _load_state_dict_flexible(model, ckpt_path, submodule_prefix_hints=()):
    import os
    if not os.path.isfile(ckpt_path):
        print(f"[WARN] ckpt not found: {ckpt_path}")
        return []

    state = torch.load(ckpt_path, map_location="cpu")
    if "state_dict" in state:  # lightning / ddp style
        state = state["state_dict"]

    # strip prefixes
    clean = {}
    for k, v in state.items():
        new_k = k
        for pf in submodule_prefix_hints:
            if new_k.startswith(pf):
                new_k = new_k[len(pf):]
        # also remove "module."
        if new_k.startswith("module."):
            new_k = new_k[7:]
        clean[new_k] = v

    missing, unexpected = model.load_state_dict(clean, strict=False)
    print(f"[LOAD] from {ckpt_path}, missing={len(missing)}, unexpected={len(unexpected)}")
    return [k for k in clean.keys() if k not in missing]


########################################
# Base Transformer (for SNP / Soma)
########################################
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
        return self.encoder(x)[:, 0]   # (B, hidden)


########################################
# ECG Transformer
########################################
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
        return self.encoder(x)[:, 0]   # (B, hidden)


########################################
# ResNet Cox
########################################
class ResNetCox(nn.Module):
    def __init__(self, arch="resnet18", freeze=False):
        super().__init__()
        backbone = timm.create_model(arch, pretrained=False, num_classes=0)
        if freeze:
            for p in backbone.parameters():
                p.requires_grad = False
        self.backbone = backbone
        self.fc = nn.Linear(backbone.num_features, 1)

    def forward(self, x):
        feat = self.backbone(x)
        return self.fc(feat).squeeze(-1)


########################################
# ViT + Cox head
########################################
class ViTHeadModel(nn.Module):
    def __init__(self, vit_name="vit_small_patch16_224", freeze=False):
        super().__init__()
        vit = timm.create_model(vit_name, pretrained=False)
        vit.reset_classifier(0)
        if freeze:
            for p in vit.parameters(): p.requires_grad = False
        self.vit = vit
        self.head = nn.Linear(vit.num_features, 1)

    def forward(self, x):
        return self.head(self.vit(x)).squeeze(-1)


########################################
# RetFound Cox Head
########################################
class RetFoundCox(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.mlp(x).squeeze(-1)


########################################
# ECG BiLSTM Cox
########################################
class BiLSTM_Cox(nn.Module):
    def __init__(self, input_size=12, hidden=128, num_layers=2, bidirectional=True):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers, batch_first=True, bidirectional=bidirectional)
        out_dim = hidden * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Linear(out_dim, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        y, _ = self.lstm(x)
        return self.head(y[:, -1, :]).squeeze(-1)


########################################
# ECG CNN Cox
########################################
class CNN1D_TextCox(nn.Module):
    def __init__(self, input_ch=12, num_filters=64, kernel_sizes=(3,5,7)):
        super().__init__()
        self.convs = nn.ModuleList([nn.Conv1d(input_ch, num_filters, k, padding=k//2) for k in kernel_sizes])
        concat = num_filters * len(kernel_sizes)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.head = nn.Sequential(nn.Linear(concat, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(self, x):
        x = x.transpose(1, 2)
        feats = [torch.relu(conv(x)) for conv in self.convs]
        z = torch.cat(feats, dim=1)
        z = self.pool(z).squeeze(-1)
        return self.head(z).squeeze(-1)


########################################
# Multi-Modal Fusion Model
########################################
class FusionModel(nn.Module):
    def __init__(self, use_fundus, use_snp, use_soma, use_ecg,
                 snp_dim=0, feature_cols=None, init_ckpts=None, use_pretrained_ckpt=True):
        super().__init__()
        self.use_fundus, self.use_snp, self.use_soma, self.use_ecg = use_fundus, use_snp, use_soma, use_ecg

        snp_hidden, snp_layers = 128, 4
        soma_hidden, soma_layers = 128, 2
        ecg_hidden, ecg_layers = 128, 3

        modules = []

        if use_fundus:
            self.fundus_enc = timm.create_model("vit_small_patch16_224", pretrained=False)
            try:
                self.fundus_enc.load_state_dict(torch.load("vit_small/pytorch_model.bin", map_location="cpu"), strict=False)
            except:
                pass
            self.fundus_enc.reset_classifier(0)
            self.fundus_proj = nn.Linear(self.fundus_enc.num_features, 128)
            modules.append(128)

        if use_snp:
            self.snp_enc = TransformerModule(snp_dim, hidden_dim=snp_hidden, nlayers=snp_layers)
            modules.append(128)

        if use_soma:
            self.soma_enc = SomaTokenEncoder(feature_cols, hidden_dim=soma_hidden, nlayers=soma_layers)
            modules.append(128)

        if use_ecg:
            self.ecg_enc = ECGTransformer(input_dim=12, hidden_dim=ecg_hidden, nlayers=ecg_layers)
            modules.append(128)

        self.attn_net = nn.Sequential(nn.Linear(128,32), nn.Tanh(), nn.Linear(32,1))
        self.mlp = nn.Sequential(nn.Linear(128,64), nn.ReLU(), nn.Linear(64,1))

        if init_ckpts:
            if use_fundus and "fundus" in init_ckpts:
                _load_state_dict_flexible(self.fundus_enc, init_ckpts["fundus"])
            if use_snp and "snp" in init_ckpts:
                _load_state_dict_flexible(self.snp_enc, init_ckpts["snp"])
            if use_soma and "soma" in init_ckpts:
                _load_state_dict_flexible(self.soma_enc, init_ckpts["soma"])
            if use_ecg and "ecg" in init_ckpts:
                _load_state_dict_flexible(self.ecg_enc, init_ckpts["ecg"])

    def extract_features(self, fundus, snp, soma, ecg):
        single = {}
        if self.use_fundus: single["fundus"] = self.fundus_proj(self.fundus_enc(fundus))
        if self.use_snp:    single["snp"]    = self.snp_enc(snp)
        if self.use_soma:   single["soma"]   = self.soma_enc(soma)
        if self.use_ecg:    single["ecg"]    = self.ecg_enc(ecg)

        feats = torch.stack(list(single.values()), dim=1)
        attn_w = torch.softmax(self.attn_net(feats), dim=1)
        fused = (feats * attn_w).sum(1)
        return single, fused   # dict of single-modal embeddings + fused embedding

    def forward(self, fundus, snp, soma, ecg):
        feats = []
        if self.use_fundus: feats.append(self.fundus_proj(self.fundus_enc(fundus)))
        if self.use_snp: feats.append(self.snp_enc(snp))
        if self.use_soma: feats.append(self.soma_enc(soma))
        if self.use_ecg: feats.append(self.ecg_enc(ecg))

        feats = torch.stack(feats, dim=1)
        attn_w = torch.softmax(self.attn_net(feats), dim=1)
        fused = (feats * attn_w).sum(1)
        return self.mlp(fused).squeeze(-1)
