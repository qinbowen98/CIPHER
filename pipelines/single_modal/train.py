# main_single.py
import os
import warnings
warnings.filterwarnings("ignore")

import argparse
from datetime import datetime
import xml.etree.ElementTree as ET

# ==== PyData ====
import numpy as np
import pandas as pd

# ==== Torch ====
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

# ==== CV ====
from PIL import Image
from torchvision import transforms

# ==== Models / Survival ====
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats import rankdata, norm
import re
# ==== PLINK ====
from pandas_plink import read_plink

# ==== Your models (keep as-is) ====
from model import (
    TransformerModule,
    ECGTransformer,
    ResNetCox,
    ViTHeadModel,
    RetFoundCox,
    BiLSTM_Cox,
    CNN1D_TextCox,
    SomaTokenEncoder,
    _load_state_dict_flexible,
)

# ==== Interpretability ====
from captum.attr import IntegratedGradients
from sklearn.model_selection import train_test_split, GroupKFold, StratifiedKFold
# ------------------------------------------------------------------------------------
# Config
# ------------------------------------------------------------------------------------
VISIT_OFFSET = {"02": 0, "07": 5, "12": 10, "20": 18}
ENDPOINTS = {
    "CHD": "fCHD_y",
    "ICVD": "fICVD_y",
    "CVDdeath": "death_yCDC",
    "stroke": "fstroke_y",
    "istroke": "fistroke_y",
    "hstroke": "fhstroke_y",
    "ICVDr": "fICVDr_y",
    "UA": "fUA_y",
}

# ------------------------------------------------------------------------------------
# Utils
# ------------------------------------------------------------------------------------
class MLP_Cox(nn.Module):
    def __init__(self, in_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x):
        return self.net(x).squeeze(-1)


class PCA_Cox(nn.Module):
    def __init__(self, pca_components=64):
        super().__init__()
        self.head = nn.Linear(pca_components, 1)
    def forward(self, x):
        return self.head(x).squeeze(-1)

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def safe_to_csv(df, path):
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)

def read_ecg_from_xml(file_path, max_len=1000):
    """Read ECG XML -> (max_len, 12) float32, pad with zeros if short."""
    try:
        if not os.path.isfile(file_path):
            return np.zeros((max_len, 12), dtype=np.float32)
        tree = ET.parse(file_path)
        root = tree.getroot()
        leads = []
        for seq in root.findall(".//sequence"):
            digits = seq.find("value/digits")
            if digits is not None and digits.text:
                leads.append(list(map(int, digits.text.strip().split())))
        leads = leads[:12]
        if len(leads) == 0:
            return np.zeros((max_len, 12), dtype=np.float32)
        min_len = min(len(ld) for ld in leads)
        arr = np.array([ld[:min_len] for ld in leads], dtype=np.float32).T  # (T, 12)
        if arr.shape[0] >= max_len:
            return arr[:max_len]
        out = np.zeros((max_len, 12), dtype=np.float32)
        out[:arr.shape[0]] = arr
        return out
    except Exception:
        return np.zeros((max_len, 12), dtype=np.float32)

class CoxHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)    # (B,)
# ------------------------------------------------------------------------------------
# Normalization
# ------------------------------------------------------------------------------------
def load_gene_list(path):
    """
    Robust reader for a gene list file. Accepts CSV/TSV/TXT with either:
      - one gene symbol per line, or
      - a header line like 'Gene' followed by entries, or
      - a single line with space/comma/semicolon separated symbols.
    Returns a list[str] of unique gene symbols (uppercase, stripped).
    """
    genes = []
    if not os.path.isfile(path):
        print(f"[WARN] Gene list file not found: {path}")
        return genes
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read().strip()
        # split by newlines first
        lines = [x.strip() for x in raw.replace("\r", "\n").split("\n") if x.strip()]
        # If only 1 line and contains many separators, split by non-word chars
        if len(lines) == 1 and ("," in lines[0] or ";" in lines[0] or " " in lines[0] or "\t" in lines[0]):
            parts = [p.strip() for p in re.split(r"[,\s;]+", lines[0]) if p.strip()]
            genes.extend(parts)
        else:
            # Possibly header like "Gene, A2M, ..." or one per line
            for ln in lines:
                # split by common separators, keep non-empty
                parts = [p.strip() for p in re.split(r"[,\s;]+", ln) if p.strip()]
                # Drop a leading header token like 'Gene' or 'GENE'
                if parts and parts[0].lower() in {"gene", "genes"} and len(parts) > 1:
                    parts = parts[1:]
                genes.extend(parts)
        # Normalize
        genes = list(dict.fromkeys([g.upper() for g in genes if g]))  # unique keep-order
    except Exception as e:
        print(f"[WARN] load_gene_list failed: {e}")
        genes = []
    return genes

def inverse_rank_normalize_vec(vec):
    ranks = rankdata(vec, method="average")
    u = (ranks - 0.5) / len(vec)
    return norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
def fit_ranknorm_global(df_soma_all,feature_cols):
    X=df_soma_all[feature_cols].values.astype(float)
    sorted_list = [np.sort(X[:,j]) for j in range(X.shape[1])]
    return sorted_list
# ------------------------------------------------------------------------------------
# Correct global rank normalization (train-fit → valid/test-transform)
# ------------------------------------------------------------------------------------
class GlobalRankNormalizer:
    def __init__(self):
        self.sorted_values = None
        self.n = None

    def fit(self, X):
        X = np.asarray(X, float)
        self.n = X.shape[0]
        self.sorted_values = [np.sort(X[:, j]) for j in range(X.shape[1])]
        return self

    def transform(self, X):
        X = np.asarray(X, float)
        X_new = np.zeros_like(X)
        for j in range(X.shape[1]):
            base = self.sorted_values[j]
            ranks = np.searchsorted(base, X[:, j], side="right")
            u = (ranks - 0.5) / len(base)
            X_new[:, j] = norm.ppf(np.clip(u, 1e-6, 1 - 1e-6))
        return X_new


def apply_soma_norm_train(df, feature_cols, method):
    df = df.copy()
    if method == "none":
        return df, None
    X = df[feature_cols].values.astype(float)

    if method == "minmax":
        scaler = MinMaxScaler().fit(X)
        df[feature_cols] = scaler.transform(X)
        return df, scaler

    elif method == "ranknorm":
        scaler = GlobalRankNormalizer().fit(X)
        df[feature_cols] = scaler.transform(X)
        return df, scaler

    else:
        raise ValueError()


def apply_soma_norm_test(df, feature_cols, method, scaler):
    df = df.copy()
    if method == "none":
        return df
    X = df[feature_cols].values.astype(float)
    df[feature_cols] = scaler.transform(X)
    return df

# ------------------------------------------------------------------------------------
# Data Loading helpers (file presence -> pid maps / tables)
# ------------------------------------------------------------------------------------
def load_fundus_dict(fundus_dir):
    d = {}
    if os.path.isdir(fundus_dir):
        for f in os.listdir(fundus_dir):
            if f.lower().endswith((".jpg", ".jpeg", ".png")) and "0_1" in f:
                pid = os.path.splitext(f)[0].split('.')[0]
                d[pid] = os.path.join(fundus_dir, f)
    return d

def load_ecg_dict(ecg_dir):
    d = {}
    if os.path.isdir(ecg_dir):
        for f in os.listdir(ecg_dir):
            if f.lower().endswith(".xml"):
                pid = os.path.splitext(f)[0]
                d[pid] = os.path.join(ecg_dir, f)
    return d

def load_snp_table(plink_prefix):
    bim, fam, bed = read_plink(plink_prefix)
    idx = fam.iid.astype(str).str.split("_").str[-1]  # fam.iid like A_B -> take last part
    snp_df = pd.DataFrame(bed.compute().T, index=idx, columns=bim.snp.astype(str)).fillna(0)
    return snp_df

def load_soma_table(path):
    # Expect SampleId: "PID" or "Vxx-PID"
    df = pd.read_csv(path)
    def parse_sid(x):
        parts = str(x).split("-")
        if len(parts) == 2:
            return parts[-1], "02"
        elif len(parts) >= 3:
            #print(parts)
            return parts[-1], parts[1]
        else:
            return str(x), "02"
    pid, visit = zip(*df["SampleId"].map(parse_sid))
    df["pid"] = list(pid)
    df["visit"] = list(visit)
    #print('load_soma_table(path)',df)
    return df

def load_shougang_test_ids(cmcs_standard_csv):
    if not os.path.isfile(cmcs_standard_csv):
        print(f"[WARN] CMCS standard CSV not found: {cmcs_standard_csv}. Test split may be empty.")
        return set()
    df = pd.read_csv(cmcs_standard_csv)
    cols = [c.lower() for c in df.columns]
    cand_id_cols = [c for c in df.columns if c.lower() in {"id", "pid", "cmcs_id", "subject_id"}]
    if len(cand_id_cols) == 0:
        raise RuntimeError("Cannot find an ID column in CMCS standard CSV.")
    id_col = cand_id_cols[0]
    if "subcohort" not in cols:
        raise RuntimeError("Cannot find 'subcohort' column in CMCS standard CSV.")
    shougang = set(df[df["subcohort"] == 102][id_col].astype(str).tolist())
    return shougang
'''
def attach_time_event(df_pid, df_label, visit_map, visit_required=True):
    df = df_pid.merge(df_label, on="pid", how="inner")
    print('merge',df)
    print(df_label['pid'][0])
    print(df_pid['pid'][0])
    if visit_required and "visit" not in df.columns:
        df["visit"] = df["pid"].map(lambda x: visit_map.get(str(x), "02"))
    df["offset"] = df["visit"].map(VISIT_OFFSET)
    df["adjusted_time"] = df["time"] - df["offset"]
    df = df[df["adjusted_time"] >= 0].copy()
    return df
'''
def attach_time_event(df_pid,df_label,modality):
    df = df_pid.merge(df_label,on='pid',how='inner')
    if modality == 'soma':
        df['offset'] = df['visit'].map(VISIT_OFFSET)
        df['adjusted_time'] = df['time'] - df['offset']
        df = df[df['adjusted_time'] >= 0].copy()
        return df
    else:
        df['offset'] = 10
        df['adjusted_time'] = df['time'] - df['offset']
        df = df[df['adjusted_time']>=0].copy()
        return df

# ------------------------------------------------------------------------------------
# Survival helpers
# ------------------------------------------------------------------------------------
def cox_ph_loss(risk, time, event):
    order = torch.argsort(-time)
    risk, event = risk[order], event[order].float()
    log_cumsum = torch.logcumsumexp(risk, dim=0)
    return -torch.sum((risk - log_cumsum) * event) / (event.sum() + 1e-8)

def run_epoch_surv(model, loader, device, optimizer=None, record_iter=False, iter_recorder=None):
    train_mode = optimizer is not None
    model.train() if train_mode else model.eval()
    total = 0.0
    risks_all, times_all, events_all = [], [], []
    with torch.set_grad_enabled(train_mode):
        for it, batch in enumerate(loader):
            x, t, e, _pid = batch
            x = x.to(device); t = t.to(device); e = e.to(device)
            out = model(x)  # (B,)
            loss = cox_ph_loss(out, t, e)
            if train_mode:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            total += loss.item() * x.size(0)
            risks_all.append(out.detach().cpu().numpy())
            times_all.extend(t.detach().cpu().numpy().tolist())
            events_all.extend(e.detach().cpu().numpy().tolist())
            if record_iter and iter_recorder is not None:
                iter_recorder.append(loss.item())
    risks_all = np.concatenate(risks_all) if len(risks_all) else np.array([])
    try:
        ci = concordance_index(np.asarray(times_all), -risks_all, np.asarray(events_all)) if len(risks_all) else 0.0
    except:
        ci = 0
    return total / max(1, len(times_all)), ci, risks_all, times_all, events_all

def fit_surv_model(model, tr_loader, va_loader, device, epochs=50, lr=1e-3, record_iter=False):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    logs = []
    it_losses = [] if record_iter else None
    best_ci = -1.0
    best_state = None
    for ep in range(1, epochs + 1):
        tr_loss, tr_ci, *_ = run_epoch_surv(model, tr_loader, device, optimizer=opt,
                                            record_iter=record_iter, iter_recorder=it_losses)
        va_loss, va_ci, va_risk, va_time, va_event = run_epoch_surv(model, va_loader, device, optimizer=None)
        logs.append({"epoch": ep, "train_loss": tr_loss, "train_ci": tr_ci, "valid_ci": va_ci})
        #print('tr_ci',tr_ci,'va_ci',va_ci)
        if va_ci > best_ci:
            best_ci = va_ci
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    df_log = pd.DataFrame(logs)
    if record_iter:
        df_log["iter_losses"] = pd.Series([np.nan]*len(df_log))
        
    return df_log, best_ci, best_state, (it_losses if record_iter else None)

def make_loader(ds, bs, shuffle):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle, num_workers=0, pin_memory=True)
class CoxHead(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, h):
        return self.fc(h).squeeze(-1)    # (B,)

# ------------------------------------------------------------------------------------
# Simple image pipeline for FundusDataset
# ------------------------------------------------------------------------------------
class FundusDataset(Dataset):
    def __init__(self, df_ids, img_map, image_size=224):
        self.df = df_ids.reset_index(drop=True)
        self.img_map = img_map

        self.tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ])

        # ---- NEW: preload images into memory ----
        self.cache_img = {}
        print(f"[Preload Fundus] {len(self.df)} samples.")
        for pid in self.df["pid"].unique():
            path = self.img_map.get(pid, None)
            if (path is None) or (not os.path.isfile(path)):
                self.cache_img[pid] = torch.zeros(3, image_size, image_size, dtype=torch.float32)
                continue
            try:
                img = Image.open(path).convert("RGB")
                self.cache_img[pid] = self.tf(img)
            except Exception as e:
                print(f"[WARN] fundus {pid} load failed: {e}")
                self.cache_img[pid] = torch.zeros(3, image_size, image_size, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = r["pid"]
        x = self.cache_img.get(pid, None)
        if x is None:
            x = torch.zeros(3, 224, 224, dtype=torch.float32)

        t = torch.tensor(r["adjusted_time"], dtype=torch.float32)
        e = torch.tensor(r["event"], dtype=torch.int64)
        return x, t, e, pid


class ECGDataset(Dataset):
    def __init__(self, df_ids, ecg_dir, max_len=1000):
        self.df = df_ids.reset_index(drop=True)
        self.ecg_dir = ecg_dir
        self.max_len = max_len

        # ---- NEW: preload ECG into memory ----
        self.cache_ecg = {}
        print(f"[Preload ECG] {len(self.df)} samples.")
        for pid in self.df["pid"].unique():
            xml_path = os.path.join(self.ecg_dir, f"{pid}.xml")
            if os.path.exists(xml_path):
                arr = read_ecg_from_xml(xml_path, max_len=max_len)
                if arr is None:
                    arr = np.zeros((max_len, 12), dtype=np.float32)
            else:
                arr = np.zeros((max_len, 12), dtype=np.float32)
            self.cache_ecg[pid] = torch.tensor(arr, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = r["pid"]
        x = self.cache_ecg.get(pid, None)
        if x is None:
            x = torch.zeros(self.max_len, 12, dtype=torch.float32)

        t = torch.tensor(r["adjusted_time"], dtype=torch.float32)
        e = torch.tensor(r["event"], dtype=torch.int64)
        return x, t, e, pid

class SNPDataset(Dataset):
    def __init__(self, df_ids, snp_table, scaler=None):
        self.df = df_ids.reset_index(drop=True)
        self.snp_table = snp_table
        self.scaler = scaler
        if scaler is not None:
            self.X = scaler.transform(snp_table.loc[self.df["pid"]].values.astype(np.float32))
        else:
            self.X = snp_table.loc[self.df["pid"]].values.astype(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = r["pid"]
        x = torch.tensor(self.X[i], dtype=torch.float32)
        t = torch.tensor(r["adjusted_time"], dtype=torch.float32)
        e = torch.tensor(r["event"], dtype=torch.int64)
        return x, t, e, pid

class SomaDataset(Dataset):
    def __init__(self, df_soma, feature_cols):
        self.df = df_soma.reset_index(drop=True)
        self.feature_cols = feature_cols
        self.X = self.df[feature_cols].values.astype(np.float32)
    def __len__(self): return len(self.df)
    def __getitem__(self, i):
        r = self.df.iloc[i]
        pid = r["pid"]
        x = torch.tensor(self.X[i], dtype=torch.float32)
        t = torch.tensor(r["adjusted_time"], dtype=torch.float32)
        e = torch.tensor(r["event"], dtype=torch.int64)
        return x, t, e, pid

# ------------------------------------------------------------------------------------
# Evaluate helper
# ------------------------------------------------------------------------------------
@torch.no_grad()
def evaluate_risk_and_embed(model, loader, device, need_embed=False):
    model.eval()
    risks, times, events, pids = [], [], [], []
    embs = []
    for batch in loader:
        x, t, e, pid = batch
        x = x.to(device)
        r = model(x).detach().cpu().numpy()
        risks.append(r)
        times.extend(t.numpy().tolist())
        events.extend(e.numpy().tolist())
        pids.extend(list(pid))
        if need_embed and hasattr(model, "embed"):
            z = model.embed(x).detach().cpu().numpy()
            embs.append(z)
    risks = np.concatenate(risks) if len(risks) else np.array([])
    ci = concordance_index(np.asarray(times), -risks, np.asarray(events)) if len(risks) else 0.0
    if need_embed and len(embs):
        embs = np.concatenate(embs, axis=0)
    else:
        embs = None
    return ci, risks, times, events, pids, embs

def save_pred_csv(outdir, tag, pids, risk, time, event):
    df = pd.DataFrame({"ID": pids, "risk": risk, "time": time, "event": event})
    safe_to_csv(df, os.path.join(outdir, "preds", f"{tag}_pred.csv"))

def save_curve_csv(outdir, tag, df_log):
    safe_to_csv(df_log, os.path.join(outdir, "curves", f"{tag}_train_curve.csv"))

def save_iter_losses(outdir, tag, iter_losses):
    if iter_losses is None: return
    df = pd.DataFrame({"iter": np.arange(len(iter_losses)), "loss": iter_losses})
    safe_to_csv(df, os.path.join(outdir, "curves", f"{tag}_iter_losses.csv"))

def save_embeddings(outdir, tag, pids, emb):
    if emb is None:
        return
    ensure_dir(os.path.join(outdir, "embeddings"))
    np.save(os.path.join(outdir, "embeddings", f"{tag}_emb.npy"),
            {"pid": np.array(pids), "emb": emb}, allow_pickle=True)

def train_evaluate_one_setting(model_builder,  # () -> nn.Module
                               ds_tr, ds_va, device, epochs, lr, bs,
                               tag_base, outdir, record_iter=False):
    model = model_builder().to(device)
    tr_loader = make_loader(ds_tr, bs=bs, shuffle=True)
    va_loader = make_loader(ds_va, bs=bs, shuffle=False)
    df_log, best_ci, best_state, it_losses = fit_surv_model(
        model, tr_loader, va_loader, device, epochs=epochs, lr=lr, record_iter=record_iter
    )
    save_curve_csv(outdir, f"{tag_base}_lr{lr}_bs{bs}", df_log)
    save_iter_losses(outdir, f"{tag_base}_lr{lr}_bs{bs}", it_losses)
    return best_ci, best_state

def select_best_setting_across_grid(results_list):
    # results_list: [(fold_id, lr, bs, best_ci, best_state), ...]
    
    df = pd.DataFrame([{"fold":f,"lr":lr,"bs":bs,"ci":ci} for (f,lr,bs,ci,_) in results_list])
    gp = df.groupby(["lr","bs"])["ci"].mean().reset_index().sort_values("ci", ascending=False)
    best_lr, best_bs, _ = gp.iloc[0][["lr","bs","ci"]]
    
    subset = [(f,lr,bs,ci,st) for (f,lr,bs,ci,st) in results_list if lr==best_lr and bs==best_bs]
    best_fold_tuple = sorted(subset, key=lambda x: x[3], reverse=True)[0]
    return float(best_lr), int(best_bs), best_fold_tuple  # (lr, bs, (fold, lr, bs, ci, state))

# ------------------------------------------------------------------------------------
# Interpretability
# ------------------------------------------------------------------------------------
def explain_integrated_gradients(model, loader, device, out_csv, feature_names=None, modality="tabular"):
    """Analysis helper. Inputs and outputs follow the function implementation."""
    model.eval()
    ig = IntegratedGradients(model)
    all_attr = []
    for batch in loader:
        x, t, e, pid = batch
        x = x.to(device)
        baseline = torch.zeros_like(x)
        attr = ig.attribute(inputs=x, baselines=baseline, target=None)
        all_attr.append(attr.detach().cpu().numpy())
    if not len(all_attr):
        return
    A = np.concatenate(all_attr, axis=0)

    if modality in ("snp", "soma"):
        mean_attr = np.mean(np.abs(A), axis=0)
        if feature_names is None:
            feature_names = [f"f{i}" for i in range(len(mean_attr))]
        pd.DataFrame({"feature": feature_names, "importance": mean_attr}).to_csv(out_csv, index=False)
    elif modality == "ecg":
        mean_attr = np.mean(np.abs(A), axis=0)  # (T, 12) if input (B,T,12)
        np.savetxt(out_csv.replace(".csv", "_ecg_attr.txt"), mean_attr, fmt="%.6f")
    else:
        mean_attr = np.mean(np.abs(A), axis=0)  # (C,H,W)
        np.save(out_csv.replace(".csv", "_fundus_ig.npy"), mean_attr)
class VitGradCAM:
    def __init__(self, vit_block):
        self.activation = None
        self.gradients = None
        vit_block.register_forward_hook(self.save_activation)
        vit_block.register_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output):
        self.activation = output
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def generate_cam(self):
        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (weights * self.activation).sum(dim=-1)  # (B, Tokens)
        cam = cam[:,1:]  # drop CLS
        return cam

def explain_vit_gradcam(vit_model, loader,device, outdir, endpoint):
    # vit_model is ViTHeadModel(vit, head)
    block = vit_model.vit.blocks[-1]
    gradcam = VitGradCAM(block)
    vit_model.eval()
    cnt = 0
    for batch in loader:
        img, t, e, pid = batch
        img = img.to(device)
        risk = vit_model(img)
        risk.mean().backward()
        cam = gradcam.generate_cam()[0].detach().cpu().numpy()
        cam_2d = cam.reshape(14, 14)  # for vit_small_patch16_224
        np.savetxt(os.path.join(outdir, f"gradcam_{cnt}_{pid[0]}_{endpoint}.txt"), cam_2d, fmt="%.6f")
'''
class VitGradCAM:
    def __init__(self, vit_block):
        self.activation = None
        self.gradients = None
        vit_block.register_forward_hook(self.save_activation)
        vit_block.register_backward_hook(self.save_gradient)
    def save_activation(self, module, input, output):
        self.activation = output
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    def generate_cam(self):
        weights = self.gradients.mean(dim=1, keepdim=True)
        cam = (weights * self.activation).sum(dim=-1)  # (B, Tokens)
        cam = cam[:, 1:]  # drop CLS
        return cam
'''
"""Analysis helper. Inputs and outputs follow the function implementation."""
'''
def explain_vit_gradcam(model, loader, device, outdir , endpoint):
    gradcam = VitGradCAM(model.fundus_enc.blocks[-1], model.fundus_enc.blocks[-1])
    samples = []
    for batch in loader:
        samples.append(batch)
        if len(samples) >= 10: break

    for i, (img, snp, soma, ecg, t, e, pid) in enumerate(samples):
        img, snp, soma, ecg = img.to(device), snp.to(device), soma.to(device), ecg.to(device)
        #img.requires_grad = True
        img_disp = img[0] if img.dim() == 4 and img.shape[0] == 1 else img
        sample_id = pid[0] if isinstance(pid,(list,tuple,np.ndarray)) else pid
        orig_img = to_pil_image((img_disp.cpu() * 0.5 + 0.5))
        orig_path = os.path.join(outdir,f"{i}_{pid[0]}_orig.png")
        orig_img.save(orig_path)

        risk = model(img, snp, soma, ecg)
        risk.mean().backward()
        cam = gradcam.generate_cam().detach().cpu().numpy()
        #print('cam',cam.shape)
        cam = gradcam.generate_cam()[0].detach().cpu().numpy()
        #print('cam',cam.shape)
        #plt.imshow(to_pil_image((img[0]*0.5+0.5).cpu()))
        plt.imshow(cam.reshape(14,14), cmap='jet', alpha=0.5)
        plt.axis('off')
        plt.savefig(f"{outdir}/gradcam_{i}_{pid[0]}_{endpoint}.png")
        plt.close()
        cam_2d = cam.reshape(14,14)
        np.savetxt(f"{outdir}/gradcam_{i}_{pid[0]}_{endpoint}.txt",cam,fmt="%.6f")
'''
def fivefold_groups_balanced(df):
    pid_event = df.groupby("pid")["event"].max()  # (pid → 0/1)
    pids = pid_event.index.values
    labels = pid_event.values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=66)
    folds = []
    for tr_pid_idx, va_pid_idx in skf.split(pids, labels):
        tr_pids = set(pids[tr_pid_idx])
        va_pids = set(pids[va_pid_idx])
        tr_idx = df.index[df["pid"].isin(tr_pids)].values
        va_idx = df.index[df["pid"].isin(va_pids)].values
        folds.append((tr_idx, va_idx))
    return folds

class SomaTokenTransformer(nn.Module):
    def __init__(self, feature_cols, hidden_dim=128, nhead=4, nlayers=2):
        super().__init__()
        self.encoder = SomaTokenEncoder(feature_cols, hidden_dim, nhead, nlayers)
        self.head = CoxHead(hidden_dim)

    def forward(self, x, return_embed=False):
        h = self.encoder(x)         # (B, D)
        risk = self.head(h)         # (B,)
        return (risk, h) if return_embed else risk
# ------------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    # Choose exactly one modality
    parser.add_argument("--fundus", action="store_true")
    parser.add_argument("--ecg", action="store_true")
    parser.add_argument("--snp", action="store_true")
    parser.add_argument("--soma", action="store_true")

    # Data paths
    parser.add_argument("--soma_train", type=str, default="Soma_origin_replicate2.csv")
    parser.add_argument("--soma_valid", type=str, default="Soma_origin_shougang_validation.csv")
    parser.add_argument("--snp_prefix", type=str, default="../CIPHER/SNP/filter_final_all")
    parser.add_argument("--fundus_dir", type=str, default="../CIPHER/Fundus/data_clean_new")
    parser.add_argument("--fundus_embed_dir", type=str, default="../CIPHER/Fundus/embeddings")  # retfound=1
    parser.add_argument("--ecg_dir", type=str, default="../CIPHER/ECG/all")
    parser.add_argument("--label_csv", type=str, default="../CIPHER/label/CMCS_outcome_final.csv")
    parser.add_argument("--cmcs_standard_csv", type=str,
                        default="data/private/CMCS_standard_21953.csv")

    # Common
    parser.add_argument("--endpoints", type=str, nargs="+", default=list(ENDPOINTS.keys()))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--outdir", type=str, default="SingleModal_Run_update_v2")

    
    parser.add_argument("--grid_lr", type=float, nargs="+", default=[1e-5, 1e-4])
    parser.add_argument("--grid_bs", type=int, nargs="+", default=[16,32])

    # Soma normalization
    parser.add_argument("--soma_norm", type=str, choices=["ranknorm", "minmax", "none","geneset"], default="ranknorm")
        # Soma gene-subset experiment
    parser.add_argument("--soma_gene_csv", type=str, default="final_genelist_Cardio.csv",
                        help="CSV/TXT of gene symbols for SOMA gene-subset experiment")
    parser.add_argument("--soma_gene_prefix", type=str, default="seq.",
                        help="Prefix of SOMA feature columns (default 'seq.' so 'seq.GENE')")
    args = parser.parse_args()
    set_seed(args.seed)

    # pick modality
    modality_flags = [args.fundus, args.ecg, args.snp, args.soma]
    if sum(map(int, modality_flags)) != 1:
        raise ValueError("Please specify exactly one modality: --fundus / --ecg / --snp / --soma")

    modality = "fundus" if args.fundus else "ecg" if args.ecg else "snp" if args.snp else "soma"

    
    exp_name = f"{modality}_{datetime.now().strftime('%Y%m%d_%H%M')}"
    OUTDIR = os.path.join(args.outdir, exp_name)
    ensure_dir(OUTDIR)
    for sub in ["data", "preds", "curves", "embeddings", "explanations", "ckpt"]:
        ensure_dir(os.path.join(OUTDIR, sub))

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # ---------- Load assets ----------
    df_soma_train = load_soma_table(args.soma_train)
    df_soma_valid = load_soma_table(args.soma_valid)
    feature_cols = [c for c in df_soma_train.columns if c.startswith("seq.")]
    fundus_dict = load_fundus_dict(args.fundus_dir)
    ecg_dict = load_ecg_dict(args.ecg_dir)
    snp_df = load_snp_table(args.snp_prefix)
    #print('fundus',len(list(set(fundus_dict.keys()))))
    #print('ecg',len(list(set(ecg_dict.keys()))))
    #print('snp',snp_df)
    #print('fundus',list(set(fundus_dict.keys()))[:10])
    #print('ecg',list(set(ecg_dict.keys()))[:10])
    df_label_all = pd.read_csv(args.label_csv)
    df_label_all["ID"] = df_label_all["ID"].astype(str)

    shougang_ids = load_shougang_test_ids(args.cmcs_standard_csv)

    has_fundus = set(fundus_dict.keys())
    has_ecg = set(ecg_dict.keys())
    has_snp = set(snp_df.index.astype(str))
    has_soma = set(df_soma_train["pid"].astype(str).tolist() + df_soma_valid["pid"].astype(str).tolist())

    def is_all_four(pid):
        pid = str(pid)
        return (pid in has_fundus) and (pid in has_ecg) and (pid in has_snp) and (pid in has_soma)

    if modality == "fundus":
        pool_train = [p for p in has_fundus if p not in shougang_ids and not is_all_four(p)]
        pool_test  = [p for p in has_fundus if p in shougang_ids and  is_all_four(p)]
        pool_rest = [p for p in has_soma if p not in shougang_ids and is_all_four(p)]
    elif modality == "ecg":
        pool_train = [p for p in has_ecg if p not in shougang_ids and not is_all_four(p)]
        pool_test  = [p for p in has_ecg if p in shougang_ids and  is_all_four(p)]
        pool_rest = [p for p in has_soma if p not in shougang_ids and is_all_four(p)]
    elif modality == "snp":
        pool_train = [p for p in has_snp if p not in shougang_ids and not is_all_four(p)]
        pool_test  = [p for p in has_snp if p in shougang_ids and  is_all_four(p)]
        pool_rest = [p for p in has_soma if p not in shougang_ids and is_all_four(p)]
    else:  # soma
        pool_train = [p for p in has_soma if p not in shougang_ids and not is_all_four(p)]
        pool_test  = [p for p in has_soma if p in shougang_ids and  is_all_four(p)]
        pool_rest = [p for p in has_soma if p not in shougang_ids and is_all_four(p)]
    safe_to_csv(pd.DataFrame({"pid": pool_train}), os.path.join(OUTDIR, "data", "train_pid_pool.csv"))
    safe_to_csv(pd.DataFrame({"pid": pool_test}), os.path.join(OUTDIR, "data", "test_pid_pool.csv"))
    safe_to_csv(pd.DataFrame({"pid": pool_rest}), os.path.join(OUTDIR, "data", "rest_pid_pool.csv"))
    # ---------- Per-endpoint loop ----------
    summary_rows = []

    for endpoint in args.endpoints:
        ranknorm_mode = args.soma_norm
        if endpoint not in ENDPOINTS:
            print(f"[WARN] Unknown endpoint {endpoint}, skip")
            continue
        time_col = ENDPOINTS[endpoint]

        df_lab = df_label_all[["ID", endpoint, time_col]].dropna().copy()
        df_lab.columns = ["pid", "event", "time"]
        df_lab["pid"] = df_lab["pid"].astype(str)
        #print(df_lab["pid"])
        tr_pid_df = pd.DataFrame({"pid": pool_train})
        te_pid_df = pd.DataFrame({"pid": pool_test})
        if modality == "soma":
            df_soma_all = pd.concat([df_soma_train,df_soma_valid],axis=0,ignore_index=True)
            df_soma_all['pid'] = df_soma_all['pid'].astype(str)
            tr_pid_df = tr_pid_df.merge(df_soma_all[['pid','visit']],on='pid',how='left')
            te_pid_df = te_pid_df.merge(df_soma_all[['pid','visit']],on='pid',how='left')
        #print(df_lab)
        df_tr = attach_time_event(tr_pid_df, df_lab, modality)
        df_te = attach_time_event(te_pid_df, df_lab, modality)
        #print('df_tr',df_tr)
        if modality != "soma":
            df_tr = df_tr.drop_duplicates(subset=["pid"]).reset_index(drop=True)
            df_te = df_te.drop_duplicates(subset=["pid"]).reset_index(drop=True)
        else:
            df_tr = df_tr.reset_index(drop=True)
            df_te = df_te.reset_index(drop=True)
            df_tr = df_tr.merge(df_soma_all[['pid','visit']+feature_cols],on=['pid','visit'],how='inner')
            df_te = df_te.merge(df_soma_all[['pid','visit']+feature_cols],on=['pid','visit'],how='inner')
        if len(df_tr) < 10:
            print(f"[{endpoint}] Not enough training samples ({len(df_tr)}). Skipping.")
            continue
        if modality == "fundus":
            
            df_tr_f = df_tr.copy().reset_index(drop=True)
            folds = fivefold_groups_balanced(df_tr)
            
            # -------------------- (A) ResNet18 --------------------
            results_res = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k = df_tr_f.iloc[tr_idx].reset_index(drop=True)
                df_va_k = df_tr_f.iloc[va_idx].reset_index(drop=True)
                ds_tr_k = FundusDataset(df_tr_k, fundus_dict)
                ds_va_k = FundusDataset(df_va_k, fundus_dict)

                def build_resnet():
                    m = ResNetCox("resnet18").to(device)
                    try:
                        _load_state_dict_flexible(m.backbone, "resnet_18/resnet18-f37072fd.pth")
                    except:
                        pass
                    return m

                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_fundus_resnet_fold{k}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_resnet, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_res.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt", f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr, best_bs, (best_fold, _, _, best_ci, best_state) = select_best_setting_across_grid(results_res)
            
            # Test with best fold
            df_tr_best = df_tr_f.iloc[folds[best_fold][0]].reset_index(drop=True)
            best_model = build_resnet().to(device)
            best_model.load_state_dict(best_state)
            ds_te = FundusDataset(df_te, fundus_dict)
            ci_te, risk_te, t_te, e_te, pid_te, _ = evaluate_risk_and_embed(
                best_model, make_loader(ds_te, int(best_bs), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_fundus_resnet_test_best", pid_te, risk_te, t_te, e_te)
            summary_rows.append({"endpoint": endpoint, "modality": "fundus", "model": f"resnet18_lr{best_lr}_bs{best_bs}_fold{best_fold}", "test_cindex": ci_te})

            # -------------------- (B) ViT small freeze --------------------
            def build_vit_freeze():
                m = ViTHeadModel("vit_small_patch16_224", freeze=True).to(device)
                try: _load_state_dict_flexible(m.vit, "vit_small/pytorch_model.bin")
                except: pass
                return m
            results_vf = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                ds_tr_k = FundusDataset(df_tr_f.iloc[tr_idx], fundus_dict)
                ds_va_k = FundusDataset(df_tr_f.iloc[va_idx], fundus_dict)
                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_fundus_vitsmallFREEZE_fold{k}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_vit_freeze, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_vf.append((k, lr, bs, best_ci, best_state))
            best_lr, best_bs, (best_fold, _, _, best_ci, best_state) = select_best_setting_across_grid(results_vf)
            best_model = build_vit_freeze().to(device); best_model.load_state_dict(best_state)
            ci_te, risk_te, t_te, e_te, pid_te, _ = evaluate_risk_and_embed(best_model, make_loader(ds_te, int(best_bs), False), device)
            save_pred_csv(OUTDIR, f"{endpoint}_fundus_vitsmallFREEZE_test_best", pid_te, risk_te, t_te, e_te)
            summary_rows.append({"endpoint": endpoint, "modality": "fundus", "model": f"vit_small_freeze_lr{best_lr}_bs{best_bs}_fold{best_fold}", "test_cindex": ci_te})
            df_tr_best = df_tr_f.iloc[folds[best_fold][0]].reset_index(drop=True)
            ds_tr_best = FundusDataset(df_tr_best, fundus_dict)
            #explain_vit_gradcam(best_model, make_loader(ds_tr_best, 1, False),device,os.path.join(OUTDIR, "explanations"), endpoint)
            # -------------------- (C) ViT small finetune --------------------
            def build_vit_tune():
                m = ViTHeadModel("vit_small_patch16_224", freeze=False).to(device)
                try: _load_state_dict_flexible(m.vit, "vit_small/pytorch_model.bin")
                except: pass
                return m
            results_vt = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                ds_tr_k = FundusDataset(df_tr_f.iloc[tr_idx], fundus_dict)
                ds_va_k = FundusDataset(df_tr_f.iloc[va_idx], fundus_dict)
                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_fundus_vitsmallFT_fold{k}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_vit_tune, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_vt.append((k, lr, bs, best_ci, best_state))
            best_lr, best_bs, (best_fold, _, _, best_ci, best_state) = select_best_setting_across_grid(results_vt)
            best_model = build_vit_tune().to(device); best_model.load_state_dict(best_state)
            ci_te, risk_te, t_te, e_te, pid_te, _ = evaluate_risk_and_embed(best_model, make_loader(ds_te, int(best_bs), False), device)
            save_pred_csv(OUTDIR, f"{endpoint}_fundus_vitsmallFT_test_best", pid_te, risk_te, t_te, e_te)
            summary_rows.append({"endpoint": endpoint, "modality": "fundus", "model": f"vit_small_finetune_lr{best_lr}_bs{best_bs}_fold{best_fold}", "test_cindex": ci_te})
            df_tr_best = df_tr_f.iloc[folds[best_fold][0]].reset_index(drop=True)
            ds_tr_best = FundusDataset(df_tr_best, fundus_dict)
            
            explain_vit_gradcam(best_model, make_loader(ds_tr_best, 1, False),device,
                                os.path.join(OUTDIR, "explanations"), endpoint)

            # -------------------- (D) RetFound 2048 MLP --------------------
            class FundusRetDS(Dataset):
                def __init__(self, df_ids, emb_dir):
                    self.df = df_ids.reset_index(drop=True); self.emb_dir = emb_dir
                def __len__(self): return len(self.df)
                def __getitem__(self, i):
                    r = self.df.iloc[i]; pid=r["pid"]; p=os.path.join(self.emb_dir,f"{pid}.pt")
                    if os.path.isfile(p):
                        vec=torch.load(p,map_location="cpu")
                        if isinstance(vec,dict) and "feat" in vec: vec=vec["feat"]
                        x=torch.tensor(vec,dtype=torch.float32).view(-1)
                    else:
                        x=torch.zeros(2048)
                    return x, torch.tensor(r["adjusted_time"],dtype=torch.float32), torch.tensor(r["event"],dtype = torch.int64), pid

            results_ret = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                ds_tr_k = FundusRetDS(df_tr_f.iloc[tr_idx], args.fundus_embed_dir)
                ds_va_k = FundusRetDS(df_tr_f.iloc[va_idx], args.fundus_embed_dir)
                def build_ret():
                    return RetFoundCox(in_dim=2048).to(device)
                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag=f"{endpoint}_fundus_retfound_fold{k}"
                        best_ci,best_state=train_evaluate_one_setting(build_ret,ds_tr_k,ds_va_k,device,args.epochs,lr,bs,tag,OUTDIR,record_iter=True)
                        results_ret.append((k,lr,bs,best_ci,best_state))
            best_lr,best_bs,(best_fold,_,_,best_ci,best_state)=select_best_setting_across_grid(results_ret)
            best_model=RetFoundCox(in_dim=2048).to(device); best_model.load_state_dict(best_state)
            ds_te_ret = FundusRetDS(df_te, args.fundus_embed_dir)
            ci_te,risk_te,t_te,e_te,pid_te,_=evaluate_risk_and_embed(best_model,make_loader(ds_te_ret,int(best_bs),False),device)
            save_pred_csv(OUTDIR,f"{endpoint}_fundus_retfound_test_best",pid_te,risk_te,t_te,e_te)
            summary_rows.append({"endpoint":endpoint,"modality":"fundus","model":f"retfound_lr{best_lr}_bs{best_bs}_fold{best_fold}","test_cindex":ci_te})
            df_sum = pd.DataFrame(summary_rows)
            safe_to_csv(df_sum, os.path.join(OUTDIR, "summary_test_cindex.csv"))
            continue

        # -------------------- ECG / SNP / SOMA: 5-fold CV (grid over lr, bs) --------------------
        # Build base tables for modality
        if modality == "ecg":
            
            folds = fivefold_groups_balanced(df_tr)

            
            def build_transformer():
                class CoxHead(nn.Module):
                    def __init__(self, in_dim):
                        super().__init__()
                        self.head = nn.Linear(in_dim, 1)
                    def forward(self, h):
                        return self.head(h).squeeze(-1)
                backbone = ECGTransformer(input_dim=12, hidden_dim=128, nhead=4, nlayers=3)
                head = CoxHead(128)
                class ECGWithHead(nn.Module):
                    def __init__(self, enc, head):
                        super().__init__()
                        self.backbone = enc
                        self.cox = head
                    def forward(self, x):
                        h = self.backbone(x)
                        return self.cox(h)
                return ECGWithHead(backbone, head)

            def build_bilstm():
                return BiLSTM_Cox()

            def build_cnn():
                return CNN1D_TextCox()

            benchmark_models = {
                "transformer_h256_l3": build_transformer,
                "bilstm": build_bilstm,
                "cnn1d": build_cnn
            }

            for model_tag, builder in benchmark_models.items():
                print(f"\n===== ECG 5-Fold Benchmarking: {model_tag} =====")
                results_grid = []  # (fold, lr, bs, best_ci, best_state)

                for k, (tr_idx, va_idx) in enumerate(folds):
                    df_tr_k = df_tr.iloc[tr_idx].reset_index(drop=True)
                    df_va_k = df_tr.iloc[va_idx].reset_index(drop=True)
                    ds_tr_k = ECGDataset(df_tr_k, args.ecg_dir)
                    ds_va_k = ECGDataset(df_va_k, args.ecg_dir)

                    for lr in args.grid_lr:
                        for bs in args.grid_bs:
                            tag = f"{endpoint}_ecg_{model_tag}_fold{k}"
                            best_ci, best_state = train_evaluate_one_setting(
                                builder, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                                tag, OUTDIR, record_iter=True
                            )
                            results_grid.append((k, lr, bs, best_ci, best_state))
                            if best_state is not None:
                                torch.save(best_state, os.path.join(OUTDIR, "ckpt", f"{tag}_lr{lr}_bs{bs}.pt"))

                
                best_lr, best_bs, (best_fold, _, _, best_ci, best_state) = select_best_setting_across_grid(results_grid)

                
                best_model = builder().to(device)
                if best_state is not None:
                    best_model.load_state_dict(best_state)
                torch.save(best_model.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_ecg_{model_tag}_best.pt"))

                
                ds_te = ECGDataset(df_te, args.ecg_dir)
                ci_te, risk_te, t_te, e_te, pid_te, _ = evaluate_risk_and_embed(
                    best_model, make_loader(ds_te, int(best_bs), False), device, need_embed=False)
                save_pred_csv(OUTDIR, f"{endpoint}_ecg_{model_tag}_test_best", pid_te, risk_te, t_te, e_te)

                summary_rows.append({
                    "endpoint": endpoint, "modality": "ecg",
                    "model": f"{model_tag}_lr{best_lr}_bs{best_bs}_fold{best_fold}",
                    "test_cindex": ci_te
                })

                
                df_tr_best = df_tr.iloc[folds[best_fold][0]].reset_index(drop=True)
                ds_tr_best = ECGDataset(df_tr_best, args.ecg_dir)
                if 'transformer' in model_tag:
                    explain_integrated_gradients(best_model, make_loader(ds_tr_best,1, False), device,
                                             os.path.join(OUTDIR, "explanations", f"{endpoint}_ecg_{model_tag}_best_IG.csv"),
                                             modality="ecg")
                df_sum = pd.DataFrame(summary_rows)
                safe_to_csv(df_sum, os.path.join(OUTDIR, "summary_test_cindex.csv"))
            continue
        if modality == "snp":
            
            folds = fivefold_groups_balanced(df_tr)

            # ===== (A) Transformer =====
            results_tr = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k = df_tr.iloc[tr_idx].reset_index(drop=True)
                df_va_k = df_tr.iloc[va_idx].reset_index(drop=True)

                sc = StandardScaler().fit(snp_df.loc[df_tr_k["pid"]].values)
                ds_tr_k = SNPDataset(df_tr_k, snp_df, scaler=sc)
                ds_va_k = SNPDataset(df_va_k, snp_df, scaler=sc)
                snptype_dim = snp_df.shape[1]

                def build_model_tr():
                    class SNPModel(nn.Module):
                        def __init__(self, in_dim=snptype_dim, hd=128, nl=4):
                            super().__init__()
                            self.enc = TransformerModule(in_dim, hidden_dim=hd, nhead=4, nlayers=nl)
                            self.head = nn.Linear(hd, 1)
                        def forward(self, x):
                            h = self.enc(x)
                            return self.head(h).squeeze(-1)
                    return SNPModel()

                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_snp_TR_fold{k}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_model_tr, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_tr.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                        f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr_tr, best_bs_tr, (best_fold_tr, _, _, best_ci_tr, best_state_tr) = \
                select_best_setting_across_grid(results_tr)
            
            df_tr_best_tr = df_tr.iloc[folds[best_fold_tr][0]].reset_index(drop=True)
            sc_best_tr = StandardScaler().fit(snp_df.loc[df_tr_best_tr["pid"]].values)
            ds_te_tr = SNPDataset(df_te, snp_df, scaler=sc_best_tr)

            class FinalSNP_TR(nn.Module):
                def __init__(self, in_dim=snp_df.shape[1], hd=128, nl=4):
                    super().__init__()
                    self.enc = TransformerModule(in_dim, hidden_dim=hd, nhead=4, nlayers=nl)
                    self.head = nn.Linear(hd, 1)
                def forward(self, x):
                    h = self.enc(x)
                    return self.head(h).squeeze(-1)

            best_model_tr = FinalSNP_TR().to(device)
            if best_state_tr is not None:
                best_model_tr.load_state_dict(best_state_tr)
            torch.save(best_model_tr.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_snp_TR_best.pt"))

            ci_te_tr, risk_te_tr, t_te_tr, e_te_tr, pid_te_tr, _ = evaluate_risk_and_embed(
                best_model_tr, make_loader(ds_te_tr, int(best_bs_tr), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_snp_TR_test_best", pid_te_tr, risk_te_tr, t_te_tr, e_te_tr)
            summary_rows.append({"endpoint": endpoint, "modality": "snp",
                                 "model": f"transformer_h256_l4_lr{best_lr_tr}_bs{best_bs_tr}_fold{best_fold_tr}",
                                 "test_cindex": ci_te_tr})

            # IG for transformer
            ds_tr_best_for_ig = SNPDataset(df_tr_best_tr, snp_df, scaler=sc_best_tr)
            explain_integrated_gradients(best_model_tr, make_loader(ds_tr_best_for_ig, int(best_bs_tr), False), device,
                                         os.path.join(OUTDIR, "explanations", f"{endpoint}_snp_TR_best_IG.csv"),
                                         feature_names=list(snp_df.columns), modality="snp")

            # ===== (B) MLP + Cox =====
            results_mlp = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k = df_tr.iloc[tr_idx].reset_index(drop=True)
                df_va_k = df_tr.iloc[va_idx].reset_index(drop=True)
                sc = StandardScaler().fit(snp_df.loc[df_tr_k["pid"]].values)
                ds_tr_mlp = SNPDataset(df_tr_k, snp_df, scaler=sc)
                ds_va_mlp = SNPDataset(df_va_k, snp_df, scaler=sc)

                def build_mlp():
                    return MLP_Cox(in_dim=snp_df.shape[1])

                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_snp_MLP_fold{k}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_mlp, ds_tr_mlp, ds_va_mlp, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_mlp.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                        f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr_mlp, best_bs_mlp, (best_fold_mlp, _, _, best_ci_mlp, best_state_mlp) = \
                select_best_setting_across_grid(results_mlp)
            df_tr_best_mlp = df_tr.iloc[folds[best_fold_mlp][0]].reset_index(drop=True)
            sc_best_mlp = StandardScaler().fit(snp_df.loc[df_tr_best_mlp["pid"]].values)
            ds_te_mlp = SNPDataset(df_te, snp_df, scaler=sc_best_mlp)

            best_model_mlp = MLP_Cox(in_dim=snp_df.shape[1]).to(device)
            if best_state_mlp is not None:
                best_model_mlp.load_state_dict(best_state_mlp)
            torch.save(best_model_mlp.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_snp_MLP_best.pt"))

            ci_te_mlp, risk_te_mlp, t_te_mlp, e_te_mlp, pid_te_mlp, _ = evaluate_risk_and_embed(
                best_model_mlp, make_loader(ds_te_mlp, int(best_bs_mlp), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_snp_MLP_test_best", pid_te_mlp, risk_te_mlp, t_te_mlp, e_te_mlp)
            summary_rows.append({"endpoint": endpoint, "modality": "snp",
                                 "model": f"mlpcox_lr{best_lr_mlp}_bs{best_bs_mlp}_fold{best_fold_mlp}",
                                 "test_cindex": ci_te_mlp})

            # ===== (C) PCA + Cox =====
            class SNP_PCA_DS(Dataset):
                def __init__(self, df_ids, X):
                    self.df = df_ids.reset_index(drop=True)
                    self.X = X
                def __len__(self): return len(self.df)
                def __getitem__(self, i):
                    r = self.df.iloc[i]
                    return torch.tensor(self.X[i], dtype=torch.float32), \
                           torch.tensor(r["adjusted_time"], dtype=torch.float32), \
                           torch.tensor(r["event"], dtype=torch.int64), \
                           r["pid"]

            results_pca = []
            pca_dim = 64
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k = df_tr.iloc[tr_idx].reset_index(drop=True)
                df_va_k = df_tr.iloc[va_idx].reset_index(drop=True)

                sc = StandardScaler().fit(snp_df.loc[df_tr_k["pid"]].values)
                X_tr = sc.transform(snp_df.loc[df_tr_k["pid"]].values)
                X_va = sc.transform(snp_df.loc[df_va_k["pid"]].values)
                pca = PCA(n_components=pca_dim).fit(X_tr)
                X_tr_pca = pca.transform(X_tr).astype(np.float32)
                X_va_pca = pca.transform(X_va).astype(np.float32)

                ds_tr_pca = SNP_PCA_DS(df_tr_k, X_tr_pca)
                ds_va_pca = SNP_PCA_DS(df_va_k, X_va_pca)

                def build_pca_model():
                    return PCA_Cox(pca_components=pca_dim)

                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_snp_PCA_fold{k}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_pca_model, ds_tr_pca, ds_va_pca, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_pca.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                        f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr_pca, best_bs_pca, (best_fold_pca, _, _, best_ci_pca, best_state_pca) = \
                select_best_setting_across_grid(results_pca)
            
            df_tr_best_pca = df_tr.iloc[folds[best_fold_pca][0]].reset_index(drop=True)
            sc_best_pca = StandardScaler().fit(snp_df.loc[df_tr_best_pca["pid"]].values)
            X_te_pca = sc_best_pca.transform(snp_df.loc[df_te["pid"]].values)
            pca_best = PCA(n_components=pca_dim).fit(sc_best_pca.transform(snp_df.loc[df_tr_best_pca["pid"]].values))
            X_te_pca = pca_best.transform(X_te_pca).astype(np.float32)
            ds_te_pca = SNP_PCA_DS(df_te, X_te_pca)

            best_model_pca = PCA_Cox(pca_components=pca_dim).to(device)
            if best_state_pca is not None:
                best_model_pca.load_state_dict(best_state_pca)
            torch.save(best_model_pca.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_snp_PCA_best.pt"))

            ci_te_pca, risk_te_pca, t_te_pca, e_te_pca, pid_te_pca, _ = evaluate_risk_and_embed(
                best_model_pca, make_loader(ds_te_pca, int(best_bs_pca), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_snp_PCA_test_best", pid_te_pca, risk_te_pca, t_te_pca, e_te_pca)
            summary_rows.append({"endpoint": endpoint, "modality": "snp",
                                 "model": f"pca{pca_dim}+cox_lr{best_lr_pca}_bs{best_bs_pca}_fold{best_fold_pca}",
                                 "test_cindex": ci_te_pca})
            df_sum = pd.DataFrame(summary_rows)
            safe_to_csv(df_sum, os.path.join(OUTDIR, "summary_test_cindex.csv"))
            continue
        if modality == "soma":
            
            folds = fivefold_groups_balanced(df_tr)

            # ===== (A) Transformer =====
            results_tr = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k_raw = df_tr.iloc[tr_idx].reset_index(drop=True)
                df_va_k_raw = df_tr.iloc[va_idx].reset_index(drop=True)
                gene_syms = []
                if ranknorm_mode == "geneset":
                    ranknorm_mode = "ranknorm"
                    all_cols = set(df_soma_train.columns)
                    print(list(all_cols)[:10],flush = True)
                    gene_syms = pd.read_csv(args.soma_gene_csv)['Gene'].tolist()
                    print(gene_syms[:10],flush=True)
                if len(gene_syms) == 0:
                    print(f"[WARN] Gene-subset skipped: empty gene list from {args.soma_gene_csv}")
                else:
                    feature_cols = [g for g in all_cols if len(g.split('.')) > 1 and g.split('.')[1] in gene_syms]
                if len(feature_cols) < 5:
                    print(f"[WARN] Gene-subset has too few matched features ({len(feature_cols)}). Skipped.")
                else:
                    print(f"[Gene-Subset] Using {len(feature_cols)} SOMA features from {args.soma_gene_csv}")
                df_tr_k, scaler_k = apply_soma_norm_train(df_tr_k_raw, feature_cols, ranknorm_mode)
                df_va_k = apply_soma_norm_test(df_va_k_raw, feature_cols, ranknorm_mode, scaler_k)

                ds_tr_k = SomaDataset(df_tr_k, feature_cols)
                ds_va_k = SomaDataset(df_va_k, feature_cols)

                def build_model_tr():
                    return SomaTokenTransformer(feature_cols, hidden_dim=128, nhead=4, nlayers=2)
                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_soma_TR_fold{k}_{args.soma_norm}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_model_tr, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_tr.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                        f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr_tr, best_bs_tr, (best_fold_tr, _, _, best_ci_tr, best_state_tr) = \
                select_best_setting_across_grid(results_tr)

            
            df_tr_best_raw_tr = df_tr.iloc[folds[best_fold_tr][0]].reset_index(drop=True)
            df_tr_best_norm_tr, scaler_best_tr = apply_soma_norm_train(df_tr_best_raw_tr, feature_cols, ranknorm_mode)
            df_te_norm_tr = apply_soma_norm_test(df_te, feature_cols, ranknorm_mode, scaler_best_tr)
            ds_te_tr = SomaDataset(df_te_norm_tr, feature_cols)
            best_model_tr = SomaTokenTransformer(feature_cols, hidden_dim=128, nhead=4, nlayers=2).to(device)

            if best_state_tr is not None:
                best_model_tr.load_state_dict(best_state_tr)
            torch.save(best_model_tr.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_soma_TR_best.pt"))

            ci_te_tr, risk_te_tr, t_te_tr, e_te_tr, pid_te_tr, _ = evaluate_risk_and_embed(
                best_model_tr, make_loader(ds_te_tr, int(best_bs_tr), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_soma_TR_test_best", pid_te_tr, risk_te_tr, t_te_tr, e_te_tr)
            summary_rows.append({"endpoint": endpoint, "modality": "soma",
                                 "model": f"transformer_{args.soma_norm}_h128_l2_lr{best_lr_tr}_bs{best_bs_tr}_fold{best_fold_tr}",
                                 "test_cindex": ci_te_tr})

            # IG（Transformer）
            ds_tr_best_ig = SomaDataset(df_tr_best_norm_tr, feature_cols)
            explain_integrated_gradients(best_model_tr, make_loader(ds_tr_best_ig, 4, False), device,
                                         os.path.join(OUTDIR, "explanations", f"{endpoint}_soma_TR_best_IG.csv"),
                                         feature_names=feature_cols, modality="soma")

            # ===== (B) MLP + Cox =====
            results_mlp = []
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k_raw = df_tr.iloc[tr_idx].reset_index(drop=True)
                df_va_k_raw = df_tr.iloc[va_idx].reset_index(drop=True)

                df_tr_k, scaler_k = apply_soma_norm_train(df_tr_k_raw, feature_cols, ranknorm_mode)
                df_va_k = apply_soma_norm_test(df_va_k_raw, feature_cols, ranknorm_mode, scaler_k)

                ds_tr_k = SomaDataset(df_tr_k, feature_cols)
                ds_va_k = SomaDataset(df_va_k, feature_cols)

                def build_mlp_soma():
                    return MLP_Cox(in_dim=len(feature_cols))

                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_soma_MLP_fold{k}_{args.soma_norm}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_mlp_soma, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_mlp.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                        f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr_mlp, best_bs_mlp, (best_fold_mlp, _, _, best_ci_mlp, best_state_mlp) = \
                select_best_setting_across_grid(results_mlp)

            df_tr_best_raw_mlp = df_tr.iloc[folds[best_fold_mlp][0]].reset_index(drop=True)
            df_tr_best_norm_mlp, scaler_best_mlp = apply_soma_norm_train(df_tr_best_raw_mlp, feature_cols, ranknorm_mode)
            df_te_norm_mlp = apply_soma_norm_test(df_te, feature_cols, ranknorm_mode, scaler_best_mlp)
            ds_te_mlp = SomaDataset(df_te_norm_mlp, feature_cols)

            best_model_mlp = MLP_Cox(in_dim=len(feature_cols)).to(device)
            if best_state_mlp is not None:
                best_model_mlp.load_state_dict(best_state_mlp)
            torch.save(best_model_mlp.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_soma_MLP_best.pt"))

            ci_te_mlp, risk_te_mlp, t_te_mlp, e_te_mlp, pid_te_mlp, _ = evaluate_risk_and_embed(
                best_model_mlp, make_loader(ds_te_mlp, int(best_bs_mlp), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_soma_MLP_test_best", pid_te_mlp, risk_te_mlp, t_te_mlp, e_te_mlp)
            summary_rows.append({"endpoint": endpoint, "modality": "soma",
                                 "model": f"mlpcox_{args.soma_norm}_lr{best_lr_mlp}_bs{best_bs_mlp}_fold{best_fold_mlp}",
                                 "test_cindex": ci_te_mlp})

            # ===== (C) PCA + Cox =====
            class SomaPCADS(Dataset):
                def __init__(self, df, X):
                    self.df = df.reset_index(drop=True)
                    self.X = X
                def __len__(self): return len(self.df)
                def __getitem__(self, i):
                    r = self.df.iloc[i]
                    return torch.tensor(self.X[i], dtype=torch.float32), \
                           torch.tensor(r["adjusted_time"], dtype=torch.float32), \
                           torch.tensor(r["event"], dtype=torch.int64), \
                           r["pid"]

            results_pca = []
            pca_dim = 64
            for k, (tr_idx, va_idx) in enumerate(folds):
                df_tr_k_raw = df_tr.iloc[tr_idx].reset_index(drop=True)
                df_va_k_raw = df_tr.iloc[va_idx].reset_index(drop=True)

                df_tr_k, scaler_k = apply_soma_norm_train(df_tr_k_raw.copy(), feature_cols, ranknorm_mode)
                df_va_k = apply_soma_norm_test(df_va_k_raw.copy(), feature_cols, ranknorm_mode, scaler_k)

                pca = PCA(n_components=pca_dim).fit(df_tr_k[feature_cols])
                X_tr_pca = pca.transform(df_tr_k[feature_cols]).astype(np.float32)
                X_va_pca = pca.transform(df_va_k[feature_cols]).astype(np.float32)

                ds_tr_pca = SomaPCADS(df_tr_k, X_tr_pca)
                ds_va_pca = SomaPCADS(df_va_k, X_va_pca)

                def build_pca_soma():
                    return PCA_Cox(pca_components=pca_dim)

                for lr in args.grid_lr:
                    for bs in args.grid_bs:
                        tag = f"{endpoint}_soma_PCA_fold{k}_{args.soma_norm}"
                        best_ci, best_state = train_evaluate_one_setting(
                            build_pca_soma, ds_tr_pca, ds_va_pca, device, args.epochs, lr, bs,
                            tag, OUTDIR, record_iter=True
                        )
                        results_pca.append((k, lr, bs, best_ci, best_state))
                        if best_state is not None:
                            torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                        f"{tag}_lr{lr}_bs{bs}.pt"))

            best_lr_pca, best_bs_pca, (best_fold_pca, _, _, best_ci_pca, best_state_pca) = \
                select_best_setting_across_grid(results_pca)

            
            df_tr_best_raw_pca = df_tr.iloc[folds[best_fold_pca][0]].reset_index(drop=True)
            df_tr_best_norm_pca, scaler_best_pca = apply_soma_norm_train(df_tr_best_raw_pca, feature_cols, ranknorm_mode)
            pca_best = PCA(n_components=pca_dim).fit(df_tr_best_norm_pca[feature_cols])
            df_te_norm_pca = apply_soma_norm_test(df_te.copy(), feature_cols, ranknorm_mode, scaler_best_pca)
            X_te_pca = pca_best.transform(df_te_norm_pca[feature_cols]).astype(np.float32)
            ds_te_pca = SomaPCADS(df_te_norm_pca, X_te_pca)
            best_model_pca = PCA_Cox(pca_components=pca_dim).to(device)
            if best_state_pca is not None:
                best_model_pca.load_state_dict(best_state_pca)
            torch.save(best_model_pca.state_dict(), os.path.join(OUTDIR, "ckpt", f"{endpoint}_soma_PCA_best.pt"))

            ci_te_pca, risk_te_pca, t_te_pca, e_te_pca, pid_te_pca, _ = evaluate_risk_and_embed(
                best_model_pca, make_loader(ds_te_pca, int(best_bs_pca), False), device, need_embed=False)
            save_pred_csv(OUTDIR, f"{endpoint}_soma_PCA_test_best", pid_te_pca, risk_te_pca, t_te_pca, e_te_pca)
            summary_rows.append({"endpoint": endpoint, "modality": "soma",
                                 "model": f"pca{pca_dim}+cox_{args.soma_norm}_lr{best_lr_pca}_bs{best_bs_pca}_fold{best_fold_pca}",
                                 "test_cindex": ci_te_pca})

            # -------------------- SOMA Gene-Subset (Transformer only) --------------------
            
            all_cols = set(df_soma_train.columns)
            gene_syms = load_gene_list(args.soma_gene_csv)
            if len(gene_syms) == 0:
                print(f"[WARN] Gene-subset skipped: empty gene list from {args.soma_gene_csv}")
            else:
                feature_cols_gene = [f"{args.soma_gene_prefix}{g}" for g in gene_syms if f"{args.soma_gene_prefix}{g}" in all_cols]
                if len(feature_cols_gene) < 5:
                    print(f"[WARN] Gene-subset has too few matched features ({len(feature_cols_gene)}). Skipped.")
                else:
                    print(f"[Gene-Subset] Using {len(feature_cols_gene)} SOMA features from {args.soma_gene_csv}")
                    # Fivefold with same normalization protocol (train-fit -> valid/test-transform)
                    results_gene_tr = []
                    for k, (tr_idx, va_idx) in enumerate(folds):
                        df_tr_k_raw = df_tr.iloc[tr_idx].reset_index(drop=True)
                        df_va_k_raw = df_tr.iloc[va_idx].reset_index(drop=True)

                        
                        
                        df_tr_k_use = df_tr_k_raw.copy()
                        df_va_k_use = df_va_k_raw.copy()

                        
                        df_tr_k_norm, scaler_k = apply_soma_norm_train(df_tr_k_use, feature_cols_gene, ranknorm_mode)
                        df_va_k_norm = apply_soma_norm_test(df_va_k_use, feature_cols_gene, ranknorm_mode, scaler_k)

                        class SomaGeneDataset(Dataset):
                            def __init__(self, df, feature_cols):
                                self.df = df.reset_index(drop=True)
                                self.feature_cols = feature_cols
                                self.X = self.df[feature_cols].values.astype(np.float32)
                            def __len__(self): return len(self.df)
                            def __getitem__(self, i):
                                r = self.df.iloc[i]
                                return (torch.tensor(self.X[i], dtype=torch.float32),
                                        torch.tensor(r["adjusted_time"], dtype=torch.float32),
                                        torch.tensor(r["event"], dtype=torch.int64),
                                        r["pid"])

                        ds_tr_k = SomaGeneDataset(df_tr_k_norm, feature_cols_gene)
                        ds_va_k = SomaGeneDataset(df_va_k_norm, feature_cols_gene)

                        def build_gene_tr():
                            return SomaTokenTransformer(feature_cols_gene, hidden_dim=128, nhead=4, nlayers=2)
                        for lr in args.grid_lr:
                            for bs in args.grid_bs:
                                tag = f"{endpoint}_somaGENE_TR_fold{k}_{args.soma_norm}"
                                best_ci, best_state = train_evaluate_one_setting(
                                    build_gene_tr, ds_tr_k, ds_va_k, device, args.epochs, lr, bs,
                                    tag, OUTDIR, record_iter=True
                                )
                                results_gene_tr.append((k, lr, bs, best_ci, best_state))
                                if best_state is not None:
                                    torch.save(best_state, os.path.join(OUTDIR, "ckpt",
                                                f"{tag}_lr{lr}_bs{bs}.pt"))

                    
                    best_lr_g, best_bs_g, (best_fold_g, _, _, best_ci_g, best_state_g) = \
                        select_best_setting_across_grid(results_gene_tr)

                    
                    df_tr_best_raw_g = df_tr.iloc[folds[best_fold_g][0]].reset_index(drop=True)
                    df_tr_best_norm_g, scaler_best_g = apply_soma_norm_train(df_tr_best_raw_g.copy(), feature_cols_gene, ranknorm_mode)
                    df_te_norm_g = apply_soma_norm_test(df_te.copy(), feature_cols_gene, ranknorm_mode, scaler_best_g)
                    class SomaGeneDatasetEval(Dataset):
                        def __init__(self, df, feature_cols):
                            self.df = df.reset_index(drop=True)
                            self.feature_cols = feature_cols
                            self.X = self.df[feature_cols].values.astype(np.float32)
                        def __len__(self): return len(self.df)
                        def __getitem__(self, i):
                            r = self.df.iloc[i]
                            return (torch.tensor(self.X[i], dtype=torch.float32),
                                    torch.tensor(r["adjusted_time"], dtype=torch.float32),
                                    torch.tensor(r["event"], dtype=torch.int64),
                                    r["pid"])

                    ds_te_g = SomaGeneDatasetEval(df_te_norm_g, feature_cols_gene)

                    best_model_g = SomaTokenTransformer(feature_cols_gene, hidden_dim=128, nhead=4, nlayers=2).to(device)
                    if best_state_g is not None:
                        best_model_g.load_state_dict(best_state_g)
                    torch.save(best_model_g.state_dict(), os.path.join(OUTDIR, "ckpt",
                                f"{endpoint}_somaGENE_TR_best.pt"))

                    ci_te_g, risk_te_g, t_te_g, e_te_g, pid_te_g, _ = evaluate_risk_and_embed(
                        best_model_g, make_loader(ds_te_g, int(best_bs_g), False), device, need_embed=False)
                    save_pred_csv(OUTDIR, f"{endpoint}_somaGENE_TR_test_best", pid_te_g, risk_te_g, t_te_g, e_te_g)
                    summary_rows.append({"endpoint": endpoint, "modality": "soma",
                                         "model": f"transformer_GENESUBSET_{args.soma_norm}_lr{best_lr_g}_bs{best_bs_g}_fold{best_fold_g}",
                                         "test_cindex": ci_te_g})

                    
                    ds_tr_best_ig_g = SomaGeneDatasetEval(df_tr_best_norm_g, feature_cols_gene)
                    explain_integrated_gradients(best_model_g, make_loader(ds_tr_best_ig_g, 4, False), device,
                                                 os.path.join(OUTDIR, "explanations", f"{endpoint}_somaGENE_TR_best_IG.csv"),
                                                 feature_names=feature_cols_gene, modality="soma")
                    # Save summary
                    df_sum = pd.DataFrame(summary_rows)
                    safe_to_csv(df_sum, os.path.join(OUTDIR, "summary_test_cindex.csv"))
                continue
    # Save summary
    df_sum = pd.DataFrame(summary_rows)
    safe_to_csv(df_sum, os.path.join(OUTDIR, "summary_test_cindex.csv"))
    print("Done. Summary saved to", os.path.join(OUTDIR, "summary_test_cindex.csv"))


if __name__ == "__main__":
    main()
