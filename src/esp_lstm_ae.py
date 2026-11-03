"""Multivariate LSTM autoencoder for ESP early-failure detection (daily data).

Amp-chart logic adapted to daily SCADA values: motor current is read together
with frequency, intake/discharge pressure, temperatures and vibration, using
per-well baselines instead of nameplate amps.

Usage:
    python src/esp_lstm_ae.py                       # 5-fold CV by well + final model
    python src/esp_lstm_ae.py --folds 3 --epochs 20

Outputs (in ../output/lstm_ae/):
    scores.csv   out-of-fold daily anomaly score, alarms, per-feature error
    events.csv   one row per failure: detected?, lead time (days)
    metrics.json summary vs. two rule baselines
    ../model/esp_lstm_ae.pt  final model trained on all wells
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "wells_data_final.csv"
OUT = ROOT / "output" / "lstm_ae"
MODEL_DIR = ROOT / "model"

SENSORS = ["CURRENT", "FREQUENCY", "PRESS_INT", "PRESS_DESC",
           "TEMP_INT", "TEMP_MOT", "VIBRATION"]
ENGINEERED = ["AMP_RATIO", "AMP_PER_HZ", "PUMP_DP"]
CONT = SENSORS + ENGINEERED
FEATURES = CONT + ["OFF"]

CURRENT_FLOOR = 20.0      # 22% of rows sit exactly here -> treated as "pump off"
PRE_DAYS, POST_DAYS = 30, 60  # exclusion zone around failures for training
BASE_LAG, BASE_REF = 30, 60   # rolling baseline: 60 days ending 30 days ago


# --------------------------------------------------------------------------- data
def load(path=DATA):
    df = pd.read_csv(path, parse_dates=["DATE"]).drop(columns=["Unnamed: 0"], errors="ignore")
    df = df.sort_values(["WELL_ID", "DATE"]).reset_index(drop=True)
    df["FAIL_EVENT"] = (df["FAILURE"] > 0).astype(int)
    return df


def engineer(df):
    df = df.copy()
    df["OFF"] = (df["CURRENT"] <= CURRENT_FLOOR).astype(float)

    # Floor / zero values that mean "no reading" rather than a real value.
    df.loc[df["OFF"] == 1, SENSORS] = np.nan
    for c in ["PRESS_INT", "PRESS_DESC"]:
        df.loc[df[c] <= 0, c] = np.nan
    for c in ["TEMP_INT", "TEMP_MOT"]:
        df.loc[df[c] <= 175, c] = np.nan
    df.loc[df["FREQUENCY"] <= 25, "FREQUENCY"] = np.nan

    g = df.groupby("WELL_ID")
    # Pseudo under/overload: current vs. the well's own 30-day running median.
    base = g["CURRENT"].transform(lambda s: s.rolling(30, min_periods=5).median())
    df["AMP_RATIO"] = df["CURRENT"] / base
    # Load per Hz: separates real load change from VSD set-point change.
    df["AMP_PER_HZ"] = df["CURRENT"] / df["FREQUENCY"]
    # Pump-developed pressure.
    df["PUMP_DP"] = df["PRESS_DESC"] - df["PRESS_INT"]

    # Mask = value is a real measurement (used to weight the loss).
    for c in CONT:
        df[f"M_{c}"] = df[c].notna().astype(float)
    df[CONT] = g[CONT].ffill()
    return df


def mark_exclusion(df):
    """1 for days within [-PRE_DAYS, +POST_DAYS] of a failure (not 'healthy')."""
    df = df.copy()
    df["T_TO_FAIL"] = np.nan
    excl = np.zeros(len(df), dtype=bool)
    for _, idx in df.groupby("WELL_ID").indices.items():
        fpos = np.flatnonzero(df["FAIL_EVENT"].values[idx])
        pos = np.arange(len(idx))
        for p in fpos:
            excl[idx[(pos >= p - PRE_DAYS) & (pos <= p + POST_DAYS)]] = True
        if len(fpos):
            # days until the next failure (for event evaluation)
            nxt = np.searchsorted(fpos, pos)
            ok = nxt < len(fpos)
            ttf = np.full(len(idx), np.nan)
            ttf[ok] = fpos[nxt[ok]] - pos[ok]
            df.loc[df.index[idx], "T_TO_FAIL"] = ttf
    df["EXCLUDED"] = excl
    return df


def _rolling_z(s):
    """Deviation from the well's own recent past (causal).

    Baseline = days [t-LAG-REF, t-LAG); with LAG = window length the model's input
    window is always compared against the period just before it. Resets naturally
    after a pump change, unlike a fixed start-of-record baseline.
    """
    ref = s.shift(BASE_LAG).rolling(BASE_REF, min_periods=20)
    med = ref.median()
    iqr = ref.quantile(0.75) - ref.quantile(0.25)
    iqr = np.maximum(iqr, 0.05 * med.abs() + 1e-3)   # floor: flat / carried-forward series
    return ((s - med) / iqr).clip(-10, 10)


def scale_per_well(df):
    """Rolling robust scaling per well; days without a baseline yet are dropped."""
    df = df.copy()
    g = df.groupby("WELL_ID")
    z = pd.concat({c: g[c].transform(_rolling_z) for c in CONT}, axis=1)
    df["HAS_BASE"] = z["CURRENT"].notna() | g.cumcount().ge(BASE_LAG + BASE_REF)
    df[CONT] = z.fillna(0.0).values
    return df


def windows(df, wells, seq, healthy_only):
    """Return X (n, seq, f), M (n, seq, f) and the row index of each window's end."""
    Xs, Ms, ends = [], [], []
    mcols = [f"M_{c}" for c in CONT] + ["OFF"]  # OFF is always observed
    for w in wells:
        sub = df[df["WELL_ID"] == w]
        if len(sub) < seq:
            continue
        x = sub[FEATURES].to_numpy(np.float32, copy=True)
        m = sub[mcols].to_numpy(np.float32, copy=True)
        m[:, -1] = 1.0
        excl = sub["EXCLUDED"].to_numpy()
        base = sub["HAS_BASE"].to_numpy()
        for t in range(seq - 1, len(sub)):
            if not base[t - seq + 1]:
                continue
            if healthy_only and excl[t - seq + 1:t + 1].any():
                continue
            Xs.append(x[t - seq + 1:t + 1]); Ms.append(m[t - seq + 1:t + 1])
            ends.append(sub.index[t])
    if not Xs:
        return None, None, np.array([], dtype=int)
    return np.stack(Xs), np.stack(Ms), np.array(ends)


# -------------------------------------------------------------------------- model
class LSTMAE(nn.Module):
    def __init__(self, n_feat, hidden=64, latent=16):
        super().__init__()
        self.enc = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.to_lat = nn.Linear(hidden, latent)
        self.dec = nn.LSTM(latent, hidden, num_layers=2, batch_first=True, dropout=0.1)
        self.out = nn.Linear(hidden, n_feat)

    def forward(self, x):
        _, (h, _) = self.enc(x)
        z = self.to_lat(h[-1])
        rep = z.unsqueeze(1).repeat(1, x.size(1), 1)
        y, _ = self.dec(rep)
        return self.out(y)


def masked_err(xh, x, m):
    return ((xh - x) ** 2) * m


def train(Xtr, Mtr, Xva, Mva, epochs, device, seed=0, verbose=True):
    torch.manual_seed(seed)
    model = LSTMAE(Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    tr = torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(Mtr))
    dl = torch.utils.data.DataLoader(tr, batch_size=256, shuffle=True)
    xv, mv = torch.from_numpy(Xva).to(device), torch.from_numpy(Mva).to(device)
    best, best_state, patience = np.inf, None, 0
    for ep in range(epochs):
        model.train()
        for xb, mb in dl:
            xb, mb = xb.to(device), mb.to(device)
            loss = masked_err(model(xb), xb, mb).sum() / mb.sum()
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            vl = (masked_err(model(xv), xv, mv).sum() / mv.sum()).item()
        if verbose:
            print(f"    epoch {ep + 1:2d}  val_loss {vl:.4f}")
        if vl < best - 1e-4:
            best, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score(model, X, M, device, recent=7):
    """Score = masked MSE over the last `recent` days of each window; also per feature."""
    model.eval()
    errs = []
    for i in range(0, len(X), 2048):
        xb = torch.from_numpy(X[i:i + 2048]).to(device)
        mb = torch.from_numpy(M[i:i + 2048]).to(device)
        e = masked_err(model(xb), xb, mb)[:, -recent:]
        mm = mb[:, -recent:]
        per_feat = e.sum(1) / mm.sum(1).clamp(min=1)
        total = e.sum((1, 2)) / mm.sum((1, 2)).clamp(min=1)
        errs.append(torch.cat([total[:, None], per_feat], 1).cpu().numpy())
    return np.concatenate(errs)


def ae_errors(df, ckpt_path=None, device=None):
    """Score every window of a prepared frame (engineer -> mark_exclusion ->
    scale_per_well) with the saved autoencoder. Returns AE_SCORE + ERR_* columns."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(ckpt_path or MODEL_DIR / "esp_lstm_ae.pt", map_location=device)
    model = LSTMAE(len(ckpt["features"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    cols = ["AE_SCORE"] + [f"ERR_{c}" for c in ckpt["features"]]
    out = pd.DataFrame(np.nan, index=df.index, columns=cols)
    X, M, ends = windows(df, df["WELL_ID"].unique(), ckpt["seq"], healthy_only=False)
    if X is not None:
        out.loc[ends, cols] = score(model, X, M, device)
    return out


# --------------------------------------------------------------------- evaluation
def persist(flag, by, k=3, n=5):
    """Alarm when >= k of the last n days are flagged (per well)."""
    return flag.groupby(by).transform(lambda s: s.rolling(n, min_periods=1).sum() >= k)


def _auc(df, col):
    from sklearn.metrics import roc_auc_score
    d = df[df[col].notna()]
    pre = (d["T_TO_FAIL"] >= 1) & (d["T_TO_FAIL"] <= PRE_DAYS)
    sel = pre | ~d["EXCLUDED"]
    return round(float(roc_auc_score(pre[sel], d.loc[sel, col])), 3)


def evaluate(df, alarm_col, score_col=None, horizon=PRE_DAYS):
    events, fa_starts, healthy_days = [], 0, 0
    for w, sub in df.groupby("WELL_ID"):
        a = sub[alarm_col].fillna(False).to_numpy(bool)
        fpos = np.flatnonzero(sub["FAIL_EVENT"].to_numpy())
        for p in fpos:
            pre = a[max(0, p - horizon):p]
            hit = np.flatnonzero(pre)
            events.append({"WELL_ID": w, "DATE": sub["DATE"].iloc[p],
                           "FAILURE": int(sub["FAILURE"].iloc[p]),
                           "detected": bool(len(hit)),
                           "lead_days": int(len(pre) - hit[0]) if len(hit) else np.nan})
        healthy = ~sub["EXCLUDED"].to_numpy()
        starts = a & ~np.r_[False, a[:-1]]
        fa_starts += int((starts & healthy).sum())
        healthy_days += int(healthy.sum())
    ev = pd.DataFrame(events)
    return ev, {
        "auc_score_1to30d_vs_healthy": _auc(df, score_col) if score_col else None,
        "events": len(ev),
        "recall_30d": round(float(ev["detected"].mean()), 3),
        "median_lead_days": float(ev["lead_days"].median()) if ev["detected"].any() else None,
        "false_alarms_per_well_year": round(fa_starts / max(healthy_days, 1) * 365, 2),
        "alarm_day_rate_healthy": round(
            float(df.loc[~df["EXCLUDED"], alarm_col].fillna(False).mean()), 4),
    }


# --------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--quantile", type=float, default=0.99)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    OUT.mkdir(parents=True, exist_ok=True); MODEL_DIR.mkdir(exist_ok=True)

    raw = load()
    df = scale_per_well(mark_exclusion(engineer(raw)))
    wells = np.asarray(df["WELL_ID"].unique(), dtype=object)
    print(f"{len(df)} rows, {len(wells)} wells, {int(df.FAIL_EVENT.sum())} failures, device={device}")

    err_cols = ["AE_SCORE"] + [f"ERR_{c}" for c in FEATURES]
    df[err_cols] = np.nan
    df["THRESH"] = np.nan

    gkf = GroupKFold(n_splits=args.folds)
    for k, (tr_i, te_i) in enumerate(gkf.split(wells, groups=wells)):
        tr_w, te_w = wells[tr_i].copy(), wells[te_i]
        rng.shuffle(tr_w)
        n_val = max(1, int(0.15 * len(tr_w)))
        va_w, tr_w = tr_w[:n_val], tr_w[n_val:]
        Xtr, Mtr, _ = windows(df, tr_w, args.seq, healthy_only=True)
        Xva, Mva, va_end = windows(df, va_w, args.seq, healthy_only=True)
        print(f"fold {k + 1}/{args.folds}: train {len(tr_w)} wells ({len(Xtr)} win), "
              f"val {len(va_w)} wells, test {len(te_w)} wells")
        model = train(Xtr, Mtr, Xva, Mva, args.epochs, device, seed=args.seed + k)
        thr = np.quantile(score(model, Xva, Mva, device)[:, 0], args.quantile)
        Xte, Mte, te_end = windows(df, te_w, args.seq, healthy_only=False)
        df.loc[te_end, err_cols] = score(model, Xte, Mte, device)
        df.loc[df["WELL_ID"].isin(te_w), "THRESH"] = thr
        print(f"    threshold (val healthy q{args.quantile}) = {thr:.4f}")

    by = df["WELL_ID"]
    df["AE_ALARM"] = persist((df["AE_SCORE"] > df["THRESH"]).astype(float), by) & df["AE_SCORE"].notna()
    # Rule baselines (amp-chart heuristics on raw values).
    raw_ratio = engineer(raw)["AMP_RATIO"]
    df["RULE_AMP_ALARM"] = persist(((raw_ratio - 1).abs() > 0.2).astype(float), by)
    df["RULE_OFF_ALARM"] = persist(df["OFF"], by)

    metrics, all_ev = {}, []
    df["RULE_AMP_SCORE"] = (raw_ratio - 1).abs()
    df["RULE_OFF_SCORE"] = df.groupby("WELL_ID")["OFF"].transform(
        lambda s: s.rolling(5, min_periods=1).mean())
    for name, col, sc in [("lstm_autoencoder", "AE_ALARM", "AE_SCORE"),
                          ("rule_amp_ratio_20pct", "RULE_AMP_ALARM", "RULE_AMP_SCORE"),
                          ("rule_pump_off", "RULE_OFF_ALARM", "RULE_OFF_SCORE")]:
        ev, m = evaluate(df, col, sc)
        metrics[name] = m
        all_ev.append(ev.assign(method=name))
        print(f"{name:22s} {m}")

    # Which variables drive the AE alarms just before failures?
    pre = df[(df["T_TO_FAIL"] >= 1) & (df["T_TO_FAIL"] <= PRE_DAYS) & df["AE_ALARM"]]
    healthy = df[~df["EXCLUDED"] & df["AE_SCORE"].notna()]
    drivers = (pre[[f"ERR_{c}" for c in FEATURES]].mean()
               / healthy[[f"ERR_{c}" for c in FEATURES]].mean()).sort_values(ascending=False)
    metrics["lstm_autoencoder"]["error_lift_before_failure"] = {
        k.replace("ERR_", ""): round(v, 2) for k, v in drivers.items()}
    print("error lift (pre-failure alarms vs healthy):\n", drivers.round(2).to_string())

    keep = ["DATE", "WELL_ID", "FAILURE", "FAIL_EVENT", "T_TO_FAIL", "EXCLUDED", "OFF",
            "THRESH", "AE_ALARM", "RULE_AMP_ALARM", "RULE_OFF_ALARM"] + err_cols
    df[keep].to_csv(OUT / "scores.csv", index=False)
    pd.concat(all_ev).to_csv(OUT / "events.csv", index=False)
    (OUT / "metrics.json").write_text(json.dumps(
        {"config": vars(args), "results": metrics}, indent=2, default=str))

    # Final model on all wells (15% held out for early stopping + threshold).
    allw = wells.copy(); rng.shuffle(allw)
    n_val = max(1, int(0.15 * len(allw)))
    Xtr, Mtr, _ = windows(df, allw[n_val:], args.seq, True)
    Xva, Mva, _ = windows(df, allw[:n_val], args.seq, True)
    print("training final model on all wells")
    model = train(Xtr, Mtr, Xva, Mva, args.epochs, device, seed=args.seed, verbose=False)
    thr = float(np.quantile(score(model, Xva, Mva, device)[:, 0], args.quantile))
    torch.save({"state_dict": model.state_dict(), "features": FEATURES, "seq": args.seq,
                "threshold": thr, "current_floor": CURRENT_FLOOR,
                "base_lag": BASE_LAG, "base_ref": BASE_REF}, MODEL_DIR / "esp_lstm_ae.pt")
    print(f"saved {MODEL_DIR / 'esp_lstm_ae.pt'} (threshold {thr:.4f}); outputs in {OUT}")


if __name__ == "__main__":
    main()
