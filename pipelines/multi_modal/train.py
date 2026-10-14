# ========================= main_single.py (FiveFold + Embeddings + IG) =========================
import warnings
warnings.filterwarnings('ignore')
import os, argparse, pickle
import numpy as np
import pandas as pd
from datetime import datetime
import xml.etree.ElementTree as ET

# Torch
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# CV / io
from PIL import Image
from torchvision import transforms

# Survival / metrics
from lifelines.utils import concordance_index

# Data utils
from pandas_plink import read_plink
from sklearn.model_selection import StratifiedKFold
from scipy.stats import norm
import timm

# Captum
from captum.attr import IntegratedGradients

# ==== Your models ====
from model import (
    TransformerModule,
    ECGTransformer,
    ResNetCox,
    ViTHeadModel,
    RetFoundCox,
    BiLSTM_Cox,
    CNN1D_TextCox,
    SomaTokenEncoder,
    FusionModel,
    _load_state_dict_flexible,
)

# ==================== CONFIG ====================
VISIT_OFFSET = {"02": 0, "07": 5, "12": 10, "20": 18}
endpoints_dict = {
    "CHD": "fCHD_y",
    "istroke": "fistroke_y",
    "hstroke": "fhstroke_y",
    "CVDdeath":"death_yCDC",
    "UA":"fUA_y"
}

parser = argparse.ArgumentParser()
parser.add_argument("--ecg", action="store_true")
parser.add_argument("--soma", action="store_true")
parser.add_argument("--snp", action="store_true")
parser.add_argument("--fundus", action="store_true")
parser.add_argument("--outdir", type=str, default="MultiModal_FiveFold_Run_update_v3")
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=64)
parser.add_argument("--longitudinal", action="store_true")
parser.add_argument("--seed", type=int, default=42)

# IG options
parser.add_argument("--ig_steps", type=int, default=32)
parser.add_argument("--ig_internal_bs", type=int, default=None)
parser.add_argument("--save_ig", action="store_true")

parser.add_argument("--soma_train", required=True)
parser.add_argument("--soma_valid", required=True)
parser.add_argument("--snp_prefix", required=True)
parser.add_argument("--fundus_dir", required=True)
parser.add_argument("--ecg_dir", required=True)
parser.add_argument("--label_csv", required=True)
parser.add_argument("--soma_gene_csv", required=True)
parser.add_argument("--endpoints", nargs="+", default=list(endpoints_dict))

args = parser.parse_args()
endpoints_dict = {ep: endpoints_dict[ep] for ep in args.endpoints}

# ==================== seeds & device ====================
seed = args.seed
torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# ==================== paths ====================
SOMA_TRAIN = args.soma_train
SOMA_VALID = args.soma_valid
SNP_PREFIX = args.snp_prefix
FUNDUS_DIR = args.fundus_dir
ECG_PATH = args.ecg_dir
LABEL_PATH = args.label_csv

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
modality_str = "_".join([m for m,f in zip(["ecg","soma","snp","fundus"],
                                          [args.ecg,args.soma,args.snp,args.fundus]) if f]) or "no_modality"
state = "longitudinal" if args.longitudinal else "visit02"
OUTDIR = os.path.join(args.outdir, f"{modality_str}_{timestamp}_{state}")
for s in ["logs","preds","features","ckpt","embeddings","explanations"]:
    os.makedirs(os.path.join(OUTDIR,s),exist_ok=True)

# ==================== Normalizer (IRN) ====================
class InverseRankNormalizer:
    def __init__(self, eps=1e-6):
        self.eps = eps
        self.sorted = None
        self.n = None
    def fit(self, X):
        X = np.asarray(X)
        self.sorted = [np.sort(X[:, j]) for j in range(X.shape[1])]
        self.n = X.shape[0]
        return self
    def transform(self, X):
        X = np.asarray(X)
        Y = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            ranks = np.searchsorted(self.sorted[j], X[:, j], side='right')
            u = np.clip((ranks - 0.5) / self.n, self.eps, 1 - self.eps)
            Y[:, j] = norm.ppf(u)
        return Y

# ==================== IO helpers ====================
def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p

def safe_to_csv(df, path):
    ensure_dir(os.path.dirname(path)); df.to_csv(path, index=False)

def read_ecg_from_xml(f):
    try:
        tree = ET.parse(f); root = tree.getroot()
        arr = []
        for seq in root.findall(".//sequence"):
            d = seq.find("value/digits")
            if d is not None and d.text:
                arr.append(list(map(int, d.text.split())))
        arr = arr[:12]
        if len(arr) == 0:
            return np.zeros((1000,12))
        L = min(len(a) for a in arr)
        mat = np.array([a[:L] for a in arr]).T[:1000]
        out = np.zeros((1000,12), dtype=float); out[:mat.shape[0]] = mat
        return out
    except:
        return np.zeros((1000,12))

# ==================== Dataset ====================
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
])

class MultiModalDataset(Dataset):
    def __init__(self, df, snp_df, fundus_dict, feat, preload=True):
        self.df = df.reset_index(drop=True)
        self.snp_df = snp_df
        self.fundus_dict = fundus_dict
        self.feat = feat
        self.cache_img = {}
        self.cache_ecg = {}
        if preload:
            if args.fundus:
                for pid, paths in fundus_dict.items():
                    try:
                        self.cache_img[pid] = transform(Image.open(paths[0]).convert("RGB"))
                    except:
                        self.cache_img[pid] = torch.zeros(3,224,224)
            if args.ecg:
                for pid in self.df['pid'].unique():
                    p = os.path.join(ECG_PATH, f"{pid}.xml")
                    self.cache_ecg[pid] = torch.tensor(read_ecg_from_xml(p), dtype=torch.float32)

    def __len__(self): return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]; pid = r["pid"]
        img  = self.cache_img.get(pid, torch.zeros(3,224,224)) if args.fundus else torch.empty(1)
        snp  = torch.tensor(self.snp_df.loc[pid].values.astype(np.float32)) if args.snp else torch.empty(1)
        soma = torch.tensor(r[self.feat].values.astype(np.float32)) if args.soma else torch.empty(1)
        ecg  = self.cache_ecg.get(pid, torch.zeros(1000,12)) if args.ecg else torch.empty(1)
        t = torch.tensor(r["adjusted_time"], dtype=torch.float32)
        e = torch.tensor(r["event"], dtype=torch.int64)
        return img, snp, soma, ecg, t, e, pid

# ==================== Survival helpers ====================
def cox_loss(risk, time, event):
    o = torch.argsort(-time)
    risk, event = risk[o], event[o]
    return -torch.sum((risk - torch.logcumsumexp(risk, 0)) * event) / (event.sum() + 1e-9)

@torch.no_grad()
def cindex(model, loader, device):
    model.eval(); R=[]; T=[]; E=[]
    for img,snp,soma,ecg,t,e,_ in loader:
        img,snp,soma,ecg = img.to(device),snp.to(device),soma.to(device),ecg.to(device)
        r = model(img,snp,soma,ecg).cpu().numpy()
        R.append(r); T.extend(t.numpy()); E.extend(e.numpy())
    R = np.concatenate(R) if len(R)>0 else np.array([])
    return concordance_index(np.asarray(T), -R, np.asarray(E))

@torch.no_grad()
def predict(model, loader, device):
    model.eval(); PID=[]; R=[]; T=[]; E=[]
    for img,snp,soma,ecg,t,e,pid in loader:
        img,snp,soma,ecg = img.to(device),snp.to(device),soma.to(device),ecg.to(device)
        r = model(img,snp,soma,ecg).cpu().numpy()
        PID.extend(pid); R.append(r); T.extend(t.numpy()); E.extend(e.numpy())
    return PID, np.concatenate(R), T, E

# ==================== Load data ====================
df_label = pd.read_csv(LABEL_PATH); df_label["ID"] = df_label["ID"].astype(str)
bim, fam, bed = read_plink(SNP_PREFIX)
snp_df = pd.DataFrame(
    bed.compute().T,
    index = fam.iid.astype(str).str.split("_").str[1],
    columns = bim.snp.astype(str)
).fillna(0)

def load_soma(soma_path):
    df = pd.read_csv(soma_path)
    def parse(x):
        p = str(x).split("-")
        return (p[-1], "02") if len(p)==1 else (p[-1], p[1])
    df["pid"], df["visit"] = zip(*df["SampleId"].map(parse))
    return df

df_soma_train = load_soma(SOMA_TRAIN)
df_soma_valid = load_soma(SOMA_VALID)
feature_cols = [c for c in df_soma_train.columns if c.startswith("seq.")]
df_gene_list = pd.read_csv(args.soma_gene_csv)['Gene'].tolist()
feature_cols = [g for g in feature_cols if len(g.split('.')) > 1 and g.split('.')[1] in df_gene_list]
fundus_dict = {f.split(".")[0]: [os.path.join(FUNDUS_DIR,f)] for f in os.listdir(FUNDUS_DIR)}
ecg_dict    = {f.split(".")[0]: [os.path.join(ECG_PATH,f)] for f in os.listdir(ECG_PATH)}

def all_four(p):
    return (p in snp_df.index) and (p in fundus_dict) and (p in ecg_dict) and (p in set(df_soma_train["pid"]) | set(df_soma_valid["pid"]))

df_label = df_label[df_label["ID"].apply(all_four)].copy()

# ==================== IG helpers ====================
def build_baselines(batch):
    """Analysis helper. Inputs and outputs follow the function implementation."""
    img, snp, soma, ecg, _, _, _ = batch
    baselines = []
    inputs = []
    # fundus
    if args.fundus:
        b0 = torch.zeros_like(img)
        baselines.append(b0); inputs.append(img)
    # snp
    if args.snp:
        b1 = torch.zeros_like(snp)
        baselines.append(b1); inputs.append(snp)
    # soma
    if args.soma:
        b2 = torch.zeros_like(soma)
        baselines.append(b2); inputs.append(soma)
    # ecg
    if args.ecg:
        b3 = torch.zeros_like(ecg)
        baselines.append(b3); inputs.append(ecg)
    return tuple(inputs), tuple(baselines)

def forward_multimodal_for_ig(model, batch):
    """Analysis helper. Inputs and outputs follow the function implementation."""

    def make_placeholder_like(x):

        return torch.zeros_like(x)


    def fn(*ig_inputs):

        pos = 0
        # fundus
        if args.fundus: img = ig_inputs[pos]; pos += 1
        else: img = torch.empty(1, device=next(model.parameters()).device)
        # snp
        if args.snp: snp = ig_inputs[pos]; pos += 1
        else: snp = torch.empty(1, device=next(model.parameters()).device)
        # soma
        if args.soma: soma = ig_inputs[pos]; pos += 1
        else: soma = torch.empty(1, device=next(model.parameters()).device)
        # ecg
        if args.ecg: ecg = ig_inputs[pos]; pos += 1
        else: ecg = torch.empty(1, device=next(model.parameters()).device)
        return model(img, snp, soma, ecg)
    return fn

def run_ig_over_loader(model, loader, endpoint_dir, feature_cols, ig_steps=32, internal_bs=None):
    """Analysis helper. Inputs and outputs follow the function implementation."""
    model.eval()


    pids_all = []

    fundus_I = []  # (B,3,224,224)
    fundus_I_avg = []
    snp_I = []     # (B, D_snp)
    soma_I = []    # (B, D_soma)
    ecg_I = []     # (B, 1000, 12)

    ig = IntegratedGradients(lambda *xs: forward_multimodal_for_ig(model, None)(*xs))

    for batch in loader:
        img, snp, soma, ecg, t, e, pid = batch

        if args.fundus: img = img.to(device).requires_grad_(True)
        if args.snp:    snp = snp.to(device).requires_grad_(True)
        if args.soma:   soma = soma.to(device).requires_grad_(True)
        if args.ecg:    ecg = ecg.to(device).requires_grad_(True)


        inputs, baselines = build_baselines((img, snp, soma, ecg, t, e, pid))
        inputs = tuple(x.to(device) for x in inputs)
        baselines = tuple(b.to(device) for b in baselines)

        # IG
        attrs = ig.attribute(
            inputs=inputs,
            baselines=baselines,
            target=None,
            n_steps=ig_steps,
            internal_batch_size=internal_bs
        )


        pos = 0
        if args.fundus:
            a_img = attrs[pos].detach().cpu().numpy()        # (B,3,224,224)
            fundus_I.append(a_img)
            fundus_I_avg.append(np.mean(np.abs(a_img), axis=1))
            pos += 1
        if args.snp:
            a_snp = attrs[pos].detach().cpu().numpy()        # (B, D_snp)
            snp_I.append(a_snp); pos += 1
        if args.soma:
            a_soma = attrs[pos].detach().cpu().numpy()       # (B, D_soma)
            soma_I.append(a_soma); pos += 1
        if args.ecg:
            a_ecg = attrs[pos].detach().cpu().numpy()        # (B, 1000, 12)
            ecg_I.append(a_ecg); pos += 1

        pids_all.extend(list(pid))


    pid_arr = np.array(pids_all)
    np.save(os.path.join(endpoint_dir, "explanations", "pids.npy"), pid_arr, allow_pickle=True)

    if args.fundus and len(fundus_I):
        F = np.concatenate(fundus_I, axis=0)
        FA = np.concatenate(fundus_I_avg, axis=0)
        np.save(os.path.join(endpoint_dir, "explanations", "fundus_ig.npy"), F)
        np.save(os.path.join(endpoint_dir, "explanations", "fundus_ig_avg.npy"), FA)

    if args.snp and len(snp_I):
        S = np.concatenate(snp_I, axis=0)
        np.save(os.path.join(endpoint_dir, "explanations", "snp_ig.npy"), S)

        mn = np.mean(np.abs(S), axis=0)
        pd.DataFrame({"feature": list(snp_df.columns), "importance": mn}).to_csv(
            os.path.join(endpoint_dir, "explanations", "snp_ig_mean.csv"), index=False
        )

    if args.soma and len(soma_I):
        A = np.concatenate(soma_I, axis=0)
        np.save(os.path.join(endpoint_dir, "explanations", "soma_ig.npy"), A)
        mn = np.mean(np.abs(A), axis=0)
        pd.DataFrame({"feature": feature_cols, "importance": mn}).to_csv(
            os.path.join(endpoint_dir, "explanations", "soma_ig_mean.csv"), index=False
        )

    if args.ecg and len(ecg_I):
        E = np.concatenate(ecg_I, axis=0)
        np.save(os.path.join(endpoint_dir, "explanations", "ecg_ig.npy"), E)

        time_mean = np.mean(np.abs(E), axis=0)      # (1000,12)
        lead_mean = np.mean(np.abs(E), axis=(0,1))  # (12,)
        np.save(os.path.join(endpoint_dir, "explanations", "ecg_ig_time_mean.npy"), time_mean)
        np.save(os.path.join(endpoint_dir, "explanations", "ecg_ig_lead_mean.npy"), lead_mean)

# ==================== Embedding extraction ====================
@torch.no_grad()
def extract_embeddings(model, loader, endpoint_dir):
    """Analysis helper. Inputs and outputs follow the function implementation."""
    model.eval()
    emb_fundus = []; emb_snp = []; emb_soma = []; emb_ecg = []; emb_fused = []
    pids = []

    for img,snp,soma,ecg,t,e,pid in loader:
        img,snp,soma,ecg = img.to(device),snp.to(device),soma.to(device),ecg.to(device)

        feats = []

        if args.fundus:
            f = model.fundus_enc(img)                 # (B, dim_fundus)
            f = model.fundus_proj(f)                  # (B, 256)
            emb_fundus.append(f.detach().cpu().numpy()); feats.append(f)
        else:
            f = None

        if args.snp:
            s = model.snp_enc(snp)                    # (B, 256)
            emb_snp.append(s.detach().cpu().numpy()); feats.append(s)
        if args.soma:
            so = model.soma_enc(soma)
            emb_soma.append(so.detach().cpu().numpy()); feats.append(so)
        if args.ecg:
            e_ = model.ecg_enc(ecg)                   # (B, 256)
            emb_ecg.append(e_.detach().cpu().numpy()); feats.append(e_)

        if len(feats) == 0:
            continue

        Z = torch.stack(feats, dim=1)                 # (B, M, 256)
        attn_w = torch.softmax(model.attn_net(Z), dim=1)  # (B,M,1)
        fused = (Z * attn_w).sum(1)                   # (B,256)
        emb_fused.append(fused.detach().cpu().numpy())
        pids.extend(list(pid))

    out_dir = os.path.join(endpoint_dir, "embeddings")
    ensure_dir(out_dir)
    pids = np.array(pids)

    if args.fundus and len(emb_fundus):
        np.save(os.path.join(out_dir, "fundus_emb.npy"), {"pid": pids, "emb": np.concatenate(emb_fundus,0)}, allow_pickle=True)
    if args.snp and len(emb_snp):
        np.save(os.path.join(out_dir, "snp_emb.npy"), {"pid": pids, "emb": np.concatenate(emb_snp,0)}, allow_pickle=True)
    if args.soma and len(emb_soma):
        np.save(os.path.join(out_dir, "soma_emb.npy"), {"pid": pids, "emb": np.concatenate(emb_soma,0)}, allow_pickle=True)
    if args.ecg and len(emb_ecg):
        np.save(os.path.join(out_dir, "ecg_emb.npy"), {"pid": pids, "emb": np.concatenate(emb_ecg,0)}, allow_pickle=True)

    if len(emb_fused):
        np.save(os.path.join(out_dir, "fused_emb.npy"), {"pid": pids, "emb": np.concatenate(emb_fused,0)}, allow_pickle=True)

# ==================== main ====================
def main():
    results = []








    for endpoint, time_col in endpoints_dict.items():
        ep_dir = os.path.join(OUTDIR, endpoint)
        for s in ["fold","test","embeddings","explanations"]:
            os.makedirs(os.path.join(ep_dir, s), exist_ok=True)

        df_lab = df_label[["ID", endpoint, time_col]].dropna().copy()
        df_lab.columns = ["pid","event","time"]

        def attach(soma, lab):
            df = soma.merge(lab, on="pid", how="inner")
            df["offset"] = df["visit"].map(VISIT_OFFSET)
            df["adjusted_time"] = df["time"] - df["offset"]
            return df[df["adjusted_time"]>=0].reset_index(drop=True)

        df_train_all = attach(df_soma_train, df_lab)
        df_test      = attach(df_soma_valid, df_lab)
        if not args.longitudinal:
            df_train_all = df_train_all[df_train_all["visit"]=="02"]
            df_test      = df_test[df_test["visit"]=="02"]

        df_train_all = df_train_all.reset_index(drop=True)
        pid_group = df_train_all.groupby("pid").first().reset_index()

        K = 5
        skf = StratifiedKFold(K, shuffle=True, random_state=args.seed)

        fold_ckpts, fold_scores, fold_scalers = [], [], []

        for fold,(tr_idx, va_idx) in enumerate(skf.split(pid_group["pid"], pid_group["event"])):
            print(f"\n===== [{endpoint}] Fold {fold+1}/{K} =====")

            train_pid = pid_group["pid"].iloc[tr_idx]
            val_pid   = pid_group["pid"].iloc[va_idx]

            df_tr = df_train_all[df_train_all["pid"].isin(train_pid)].copy()
            df_va = df_train_all[df_train_all["pid"].isin(val_pid)].copy()

            # IRN on SOMA
            scaler = None
            if args.soma:
                scaler = InverseRankNormalizer().fit(df_tr[feature_cols])
                df_tr.loc[:,feature_cols] = scaler.transform(df_tr[feature_cols])
                df_va.loc[:,feature_cols] = scaler.transform(df_va[feature_cols])
                #df_test.loc[:,feature_cols] = scaler.transform(df_test[feature_cols])
                df_tr.to_csv(os.path.join(ep_dir, "fold", f"fold{fold}_train_set.csv"))
                df_va.to_csv(os.path.join(ep_dir, "fold", f"fold{fold}_valid_set.csv"))
                #df_test.to_csv(os.path.join(ep_dir, "fold", f"fold{fold}_test_set.pt"))
            ds_tr = MultiModalDataset(df_tr, snp_df, fundus_dict, feature_cols)
            ds_va = MultiModalDataset(df_va, snp_df, fundus_dict, feature_cols)

            tr_loader = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True,  num_workers=0)
            va_loader = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=0)

            model = FusionModel(
                use_fundus=args.fundus, use_snp=args.snp, use_soma=args.soma, use_ecg=args.ecg,
                snp_dim=snp_df.shape[1], feature_cols=feature_cols
            ).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
            best_ci = -1; best_state = None; patience = 0

            for epoch in range(1, args.epochs+1):
                model.train()
                for img,snp,soma,ecg,t,e,_ in tr_loader:
                    img,snp,soma,ecg,t,e = img.to(device),snp.to(device),soma.to(device),ecg.to(device),t.to(device),e.to(device)
                    loss = cox_loss(model(img,snp,soma,ecg), t, e)
                    optimizer.zero_grad(); loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                va_ci = cindex(model, va_loader, device)
                if va_ci > best_ci:
                    best_ci = va_ci
                    best_state = {k: v.detach().cpu().clone() for k,v in model.state_dict().items()}
                    patience = 0
                else:
                    patience += 1
                    if patience > 8: break


            ckpt_path = os.path.join(ep_dir, "fold", f"fold{fold}_best.pt")
            torch.save(best_state, ckpt_path)
            fold_ckpts.append(ckpt_path)
            fold_scores.append(best_ci)
            with open(os.path.join(ep_dir, "fold", f"fold{fold}_scaler.pkl"), "wb") as f:
                pickle.dump(scaler, f)


        best_fold = int(np.argmax(fold_scores))
        best_ckpt = fold_ckpts[best_fold]
        scaler_path = os.path.join(ep_dir, "fold", f"fold{best_fold}_scaler.pkl")
        scaler_best = pickle.load(open(scaler_path, "rb"))

        print(f"\n[Select Fold] endpoint={endpoint} best fold = {best_fold}, valid CI = {fold_scores[best_fold]:.4f}")

        # Test

        df_test_scaled = df_test.copy()
        if args.soma and scaler_best is not None:
            df_test_scaled.loc[:, feature_cols] = scaler_best.transform(df_test_scaled[feature_cols])
            df_test_scaled.to_csv(os.path.join(ep_dir, "test", "testset.csv"))
        ds_te = MultiModalDataset(df_test_scaled, snp_df, fundus_dict, feature_cols)
        te_loader = DataLoader(ds_te, batch_size=args.batch_size, shuffle=False, num_workers=0)

        model = FusionModel(
            use_fundus=args.fundus, use_snp=args.snp, use_soma=args.soma, use_ecg=args.ecg,
            snp_dim=snp_df.shape[1], feature_cols=feature_cols
        ).to(device)
        model.load_state_dict(torch.load(best_ckpt, map_location=device))

        PID, R, T, E = predict(model, te_loader, device)
        test_ci = concordance_index(np.asarray(T), -np.asarray(R), np.asarray(E))
        safe_to_csv(pd.DataFrame({"ID":PID, "risk":R, "time":T, "event":E}), os.path.join(ep_dir, "test", "risk_test.csv"))

        results.append((endpoint, float(np.max(fold_scores)), float(test_ci)))

        # ---- Embeddings (best model on test) ----
        extract_embeddings(model, te_loader, ep_dir)

        # ---- IG (best model on test) ----
        if args.save_ig:
            run_ig_over_loader(
                model, te_loader, ep_dir, feature_cols,
                ig_steps=args.ig_steps, internal_bs=args.ig_internal_bs
            )

        pd.DataFrame(results, columns=["endpoint","best_valid_cindex","test_cindex"]).to_csv(
                os.path.join(OUTDIR,"summary.csv"), index=False)
    print("== DONE ==")

if __name__ == "__main__":
    main()
