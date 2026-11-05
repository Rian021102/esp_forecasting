"""Supervised "failure within the next 30 days" model for ESP wells (daily data).

Two models, evaluated with the same well-grouped folds as esp_lstm_ae.py:
  * LightGBM on causal rolling features (amp-chart style deviations, shutdown /
    cycling counts, run age, well-test context, LSTM-AE reconstruction errors)
  * LSTM classifier on the raw 30-day scaled sequences

Label: y = 1 on days 1..30 before a failure; y = 0 on healthy days (outside
[-30, +60] of any failure). The failure day and the 60 days after it (workover,
restart) are not trained or scored as negatives.

Run esp_lstm_ae.py first (its out-of-fold errors in output/lstm_ae/scores.csv
are used as features). Then:
    python src/esp_failure_clf.py
Outputs: output/failure_clf/{predictions.csv, events.csv, metrics.json,
feature_importance.csv}, model/esp_failure_lgbm.txt
"""
import argparse
import json

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

import esp_ampcard as ac
import esp_lstm_ae as ae

OUT = ae.ROOT / "output" / "failure_clf"
HORIZON = ae.PRE_DAYS  # 30 days


# ----------------------------------------------------------------------- features
def build_features(df):
    """Causal per-well features from the rolling-scaled frame (one row per day)."""
    g = df.groupby("WELL_ID")
    f = pd.DataFrame(index=df.index)
    nob = ~df["HAS_BASE"]
    for c in ae.CONT:
        z = df[c].mask(nob)
        gz = z.groupby(df["WELL_ID"])
        f[f"{c}_z"] = z
        f[f"{c}_z_mean7"] = gz.transform(lambda s: s.rolling(7, min_periods=4).mean())
        f[f"{c}_z_std7"] = gz.transform(lambda s: s.rolling(7, min_periods=4).std())
        f[f"{c}_z_trend7"] = f[f"{c}_z_mean7"] - gz.transform(
            lambda s: s.shift(7).rolling(7, min_periods=4).mean())

    off = df["OFF"]
    go = off.groupby(df["WELL_ID"])
    starts = (off.diff().fillna(0) > 0).astype(float).groupby(df["WELL_ID"])
    f["off_frac7"] = go.transform(lambda s: s.rolling(7, min_periods=1).mean())
    f["off_frac30"] = go.transform(lambda s: s.rolling(30, min_periods=1).mean())
    # number of shutdowns in 30 days = daily proxy for amp-chart "cycling"
    f["off_starts30"] = starts.transform(lambda s: s.rolling(30, min_periods=1).sum())
    f["days_since_off"] = _days_since(off.astype(bool), df["WELL_ID"])

    # ESP run age: days since last failure (or start of record)
    f["run_age"] = _days_since(df["FAIL_EVENT"].astype(bool), df["WELL_ID"], start_at_zero=True)

    # Well-test context (carried forward between tests)
    for c in ["BSW", "GLR(SCF/BFPD)", "BFPD"]:
        f[c] = df[c]
        f[f"{c}_chg30"] = df[c] - g[c].shift(30)
    f["pip_test_chg30"] = df["PIP(PSI)"] - g["PIP(PSI)"].shift(30)
    return f


def _days_since(event, wells, start_at_zero=False):
    out = np.full(len(event), np.nan)
    for _, idx in wells.groupby(wells).indices.items():
        e = event.to_numpy()[idx]
        last, vals = (0 if start_at_zero else None), []
        for i, flag in enumerate(e):
            if flag:
                last = i
            vals.append(np.nan if last is None else i - last)
        out[idx] = vals
    return pd.Series(out, index=event.index)


def add_ae_errors(df, f):
    p = ae.OUT / "scores.csv"
    if not p.exists():
        print("note: output/lstm_ae/scores.csv not found; AE features skipped")
        return f
    s = pd.read_csv(p, parse_dates=["DATE"])
    cols = ["AE_SCORE"] + [c for c in s.columns if c.startswith("ERR_")]
    m = df[["WELL_ID", "DATE"]].merge(s[["WELL_ID", "DATE"] + cols], how="left",
                                      on=["WELL_ID", "DATE"])
    f[cols] = m[cols].to_numpy()
    return f


# ---------------------------------------------------------------------- inference
META_PATH = ae.MODEL_DIR / "esp_failure_lgbm.json"


# Amp-card class -> feature slug (fixed list keeps feature names stable).
CARD_SLUGS = {
    "Cycling": "cycling", "Recent trip": "recent_trip",
    "Restart after shutdown": "restart", "Extended shutdown": "extended_off",
    "Gassy / erratic loading": "erratic", "Speed change (VSD)": "speed_change",
    "High amperage (overload) – gradual rise": "overload_gradual",
    "High amperage (overload) – sudden rise": "overload_sudden",
    "High amperage (overload) – steady above baseline": "overload_steady",
    "Normal operation – steady at baseline": "normal_steady",
    "Normal operation – recovering to baseline": "normal_recovering",
}
CARD_COUNT_SLUGS = ["cycling", "recent_trip", "erratic", "overload_gradual", "underload",
                    "speed_change"]
CAUSE_SLUGS = {"Pumped off": "cause_pumpoff", "Pump-off": "cause_pumpoff",
               "Inside the pump": "cause_inside_pump", "Gas interference": "cause_gas",
               "Heavier fluid": "cause_fluid", "Mechanical drag": "cause_mechanical"}


def _card_slug(detail):
    if not isinstance(detail, str):
        return "none"
    if detail.startswith("Low amperage"):
        return "underload"
    return CARD_SLUGS.get(detail, "other")


def ampcard_features(df, cards):
    """Daily amp-card features aligned to df (one-hot class, 30-day counts, causes)."""
    m = df[["WELL_ID", "DATE"]].merge(
        cards[["WELL_ID", "DATE", "AMP_CARD_DETAIL", "LIKELY_CAUSE", "UNUSUAL", "AE_REC_ERR"]],
        on=["WELL_ID", "DATE"], how="left")
    m.index = df.index
    slug = m["AMP_CARD_DETAIL"].map(_card_slug)
    f = pd.DataFrame(index=df.index)
    for sl in sorted(set(CARD_SLUGS.values()) | {"underload", "other", "none"}):
        f[f"card_{sl}"] = (slug == sl).astype(float)
    wells = df["WELL_ID"]
    for sl in CARD_COUNT_SLUGS:
        f[f"card_{sl}_days30"] = f[f"card_{sl}"].groupby(wells).transform(
            lambda x: x.rolling(30, min_periods=1).sum())
    unusual = m["UNUSUAL"].astype("boolean").fillna(False).astype(float)
    f["card_unusual_days30"] = unusual.groupby(wells).transform(
        lambda x: x.rolling(30, min_periods=1).sum())
    f["card_rec_err"] = m["AE_REC_ERR"].astype(float)
    cause = m["LIKELY_CAUSE"].fillna("")
    for key, col in CAUSE_SLUGS.items():
        f[col] = np.maximum(f.get(col, 0.0), cause.str.startswith(key).astype(float))
    return f


# Feature sets for ablations (--features). "current" keeps only what an amp-card
# reader sees: current itself, current vs its own baseline, pump-off/shutdown
# history (current <= 20 A) and the amp-card classes. It leaves out frequency-based
# AMP_PER_HZ, the multi-signal autoencoder errors and the likely-cause flags (those
# use intake pressure, GLR, water cut and vibration).
CURRENT_PREFIXES = ("CURRENT_", "AMP_RATIO_", "off_", "days_since_off", "card_")


def select_features(cols, which):
    if which == "all":
        return list(cols)
    keep = [c for c in cols if c.startswith(CURRENT_PREFIXES)]
    if which == "current_runage":
        keep.append("run_age")
    return keep


def prepare(raw, use_ampcard=None):
    """Raw well frame (wells_data_final.csv schema) -> prepared frame + features.

    FAILURE is optional; when present, past failures reset the run-age feature.
    """
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    if "FAILURE" not in raw:
        raw["FAILURE"] = 0
    raw = raw.drop(columns=["Unnamed: 0"], errors="ignore")
    raw = raw.sort_values(["WELL_ID", "DATE"]).reset_index(drop=True)
    raw["FAIL_EVENT"] = (raw["FAILURE"] > 0).astype(int)
    df = ae.scale_per_well(ae.mark_exclusion(ae.engineer(raw)))
    F = build_features(df)
    F = pd.concat([F, ae.ae_errors(df)], axis=1)
    if use_ampcard is None:
        use_ampcard = load_meta().get("uses_ampcard", False)
    card_out = None
    if use_ampcard:
        card_out = ac.classify(raw)
        F = pd.concat([F, ampcard_features(df, card_out[0])], axis=1)
    return df, F, card_out


def load_meta():
    return json.loads(META_PATH.read_text())


def alarm(risk, wells, threshold, meta=None):
    meta = meta or load_meta()
    return ae.persist((risk > threshold).astype(float), wells,
                      k=meta["persist_k"], n=meta["persist_n"])


def score_wells(raw, threshold=None, explain=False, return_cards=False):
    """Daily 30-day failure risk per well with the saved LightGBM model.

    Returns the prepared frame with RISK, ALARM and THRESHOLD columns (RISK is a
    class-weighted score, not a calibrated probability). With explain=True also
    returns per-day feature contributions (log-odds, LightGBM pred_contrib).
    """
    meta = load_meta()
    booster = lgb.Booster(model_file=str(ae.MODEL_DIR / "esp_failure_lgbm.txt"))
    df, F, card_out = prepare(raw, meta.get("uses_ampcard", False))
    X = F[meta["features"]]
    df["RISK"] = booster.predict(X)
    thr = meta["threshold"] if threshold is None else threshold
    df["THRESHOLD"] = thr
    df["ALARM"] = alarm(df["RISK"], df["WELL_ID"], thr, meta)
    out = [df]
    if explain:
        out.append(pd.DataFrame(booster.predict(X, pred_contrib=True)[:, :-1],
                                index=df.index, columns=meta["features"]))
    if return_cards:
        out.append(card_out)   # (cards, X, ckpt) from esp_ampcard.classify, or None
    return out[0] if len(out) == 1 else tuple(out)


# ---------------------------------------------------------------- LSTM classifier
class LSTMClf(nn.Module):
    def __init__(self, n_feat, hidden=64):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=2, batch_first=True, dropout=0.2)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.ReLU(), nn.Dropout(0.2),
                                  nn.Linear(32, 1))

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.head(h[-1]).squeeze(-1)


def seq_data(df, wells, seq, labelled_only):
    X, _, ends = ae.windows(df, wells, seq, healthy_only=False)
    if X is None:
        return None, None, ends
    if labelled_only:
        keep = df.loc[ends, "Y"].notna().to_numpy()
        X, ends = X[keep], ends[keep]
    return X, df.loc[ends, "Y"].to_numpy(np.float32), ends


def train_lstm(Xtr, ytr, Xva, yva, device, epochs, seed):
    torch.manual_seed(seed)
    model = LSTMClf(Xtr.shape[2]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    pw = torch.tensor((ytr == 0).sum() / max((ytr == 1).sum(), 1), device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=256, shuffle=True)
    best, state, patience = -1, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            loss = lossf(model(xb), yb)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        ap = average_precision_score(yva, predict_lstm(model, Xva, device))
        if ap > best + 1e-4:
            best, patience = ap, 0
            state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= 5:
                break
    model.load_state_dict(state)
    return model


@torch.no_grad()
def predict_lstm(model, X, device):
    model.eval()
    out = [torch.sigmoid(model(torch.from_numpy(X[i:i + 4096]).to(device))).cpu().numpy()
           for i in range(0, len(X), 4096)]
    return np.concatenate(out)


# --------------------------------------------------------------------------- main
# Strongly regularised; tree count fixed (OOF AUC is flat from 50-400 trees and
# early stopping on ~11 validation wells was too noisy - it stopped at 1-2 trees).
LGB_PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=7,
                  min_child_samples=500, subsample=0.7, subsample_freq=1,
                  colsample_bytree=0.5, reg_lambda=10.0, n_estimators=150, verbose=-1)


def day_metrics(df, col):
    d = df[df["Y"].notna() & df[col].notna()]
    return {"auc": round(float(roc_auc_score(d["Y"], d[col])), 3),
            "pr_auc": round(float(average_precision_score(d["Y"], d[col])), 3),
            "base_rate": round(float(d["Y"].mean()), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seq", type=int, default=30)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--alarm_quantile", type=float, default=0.98,
                    help="threshold = this quantile of healthy-day probability on val wells")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no_ampcard", action="store_true",
                    help="leave out the amp-card features (ablation)")
    ap.add_argument("--features", choices=["all", "current", "current_runage"],
                    default="all", help="feature set (ablation); non-default needs --tag")
    ap.add_argument("--no_lstm", action="store_true",
                    help="skip the LSTM classifier comparison (faster)")
    ap.add_argument("--tag", default="",
                    help="write outputs to output/failure_clf_<tag> and do not save models")
    args = ap.parse_args()
    if args.features != "all" and not args.tag:
        ap.error("--features other than 'all' must be run with --tag (the app expects all features)")
    out = OUT if not args.tag else OUT.parent / f"failure_clf_{args.tag}"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed)
    out.mkdir(parents=True, exist_ok=True)

    raw = ae.load()
    df = ae.scale_per_well(ae.mark_exclusion(ae.engineer(raw)))
    pre = (df["T_TO_FAIL"] >= 1) & (df["T_TO_FAIL"] <= HORIZON)
    df["Y"] = np.where(pre, 1.0, np.where(~df["EXCLUDED"], 0.0, np.nan))
    F = add_ae_errors(df, build_features(df))
    if not args.no_ampcard:
        F = pd.concat([F, ampcard_features(df, ac.classify(raw)[0])], axis=1)
    F = F[select_features(F.columns, args.features)]
    feats = list(F.columns)
    print(f"{len(df)} rows | labelled {int(df.Y.notna().sum())} "
          f"(pos {int(df.Y.sum())}) | {len(feats)} features | device={device}")

    wells = np.asarray(df["WELL_ID"].unique(), dtype=object)
    for c in ["P_LGB", "P_LSTM", "THR_LGB", "THR_LSTM"]:
        df[c] = np.nan
    imps = []

    for k, (tr_i, te_i) in enumerate(GroupKFold(args.folds).split(wells, groups=wells)):
        tr_w, te_w = wells[tr_i].copy(), wells[te_i]
        rng.shuffle(tr_w)
        n_val = max(1, int(0.15 * len(tr_w)))
        va_w, tr_w = tr_w[:n_val], tr_w[n_val:]
        is_tr = df["WELL_ID"].isin(tr_w) & df["Y"].notna()
        is_va = df["WELL_ID"].isin(va_w) & df["Y"].notna()
        is_te = df["WELL_ID"].isin(te_w)

        # --- LightGBM
        ytr = df.loc[is_tr, "Y"]
        clf = lgb.LGBMClassifier(**LGB_PARAMS, random_state=args.seed + k,
                                 scale_pos_weight=float((ytr == 0).sum() / ytr.sum()))
        clf.fit(F[is_tr], ytr)
        pva = clf.predict_proba(F[is_va])[:, 1]
        thr = np.quantile(pva[df.loc[is_va, "Y"].to_numpy() == 0], args.alarm_quantile)
        df.loc[is_te, "P_LGB"] = clf.predict_proba(F[is_te])[:, 1]
        df.loc[is_te, "THR_LGB"] = thr
        imps.append(pd.Series(clf.booster_.feature_importance("gain"), index=feats))

        if args.no_lstm:
            print(f"fold {k + 1}/{args.folds}: lgb val PR-AUC "
                  f"{average_precision_score(df.loc[is_va, 'Y'], pva):.3f}")
            continue

        # --- LSTM classifier
        Xtr, ytr_s, _ = seq_data(df, tr_w, args.seq, True)
        Xva, yva_s, _ = seq_data(df, va_w, args.seq, True)
        lstm = train_lstm(Xtr, ytr_s, Xva, yva_s, device, args.epochs, args.seed + k)
        pv = predict_lstm(lstm, Xva, device)
        thr_l = np.quantile(pv[yva_s == 0], args.alarm_quantile)
        Xte, _, te_end = seq_data(df, te_w, args.seq, False)
        df.loc[te_end, "P_LSTM"] = predict_lstm(lstm, Xte, device)
        df.loc[is_te, "THR_LSTM"] = thr_l
        print(f"fold {k + 1}/{args.folds}: lgb val PR-AUC "
              f"{average_precision_score(df.loc[is_va, 'Y'], pva):.3f}, "
              f"lstm val PR-AUC {average_precision_score(yva_s, pv):.3f}")

    by = df["WELL_ID"]
    df["LGB_ALARM"] = ae.persist((df["P_LGB"] > df["THR_LGB"]).astype(float), by)
    df["LSTM_ALARM"] = ae.persist((df["P_LSTM"] > df["THR_LSTM"]).astype(float), by) \
        & df["P_LSTM"].notna()

    # Reference: autoencoder and amp-ratio rule from the previous step.
    s = pd.read_csv(ae.OUT / "scores.csv", parse_dates=["DATE"])
    df = df.merge(s[["WELL_ID", "DATE", "AE_SCORE", "AE_ALARM", "RULE_AMP_ALARM"]],
                  on=["WELL_ID", "DATE"], how="left")
    df["AMP_DEV"] = (ae.engineer(ae.load())["AMP_RATIO"] - 1).abs().to_numpy()

    metrics, evs = {}, []
    methods = [("lightgbm", "LGB_ALARM", "P_LGB"),
               ("lstm_classifier", "LSTM_ALARM", "P_LSTM"),
               ("lstm_autoencoder", "AE_ALARM", "AE_SCORE"),
               ("rule_amp_ratio_20pct", "RULE_AMP_ALARM", "AMP_DEV")]
    if args.no_lstm:
        methods.pop(1)
    for name, alarm, score in methods:
        df[alarm] = df[alarm].astype("boolean").fillna(False).astype(bool)
        ev, m = ae.evaluate(df, alarm)
        m.pop("auc_score_1to30d_vs_healthy", None)
        m = {**day_metrics(df, score), **m}
        metrics[name] = m
        evs.append(ev.assign(method=name))
        print(f"{name:21s} {m}")

    imp = pd.concat(imps, axis=1).mean(axis=1).sort_values(ascending=False)
    imp = (imp / imp.sum()).rename("gain_share")
    imp.to_csv(out / "feature_importance.csv")
    print("top features:\n", imp.head(15).round(3).to_string())

    df[["DATE", "WELL_ID", "FAILURE", "T_TO_FAIL", "Y", "P_LGB", "THR_LGB", "LGB_ALARM",
        "P_LSTM", "THR_LSTM", "LSTM_ALARM"]].to_csv(out / "predictions.csv", index=False)
    pd.concat(evs).to_csv(out / "events.csv", index=False)
    (out / "metrics.json").write_text(json.dumps(
        {"config": vars(args), "results": metrics}, indent=2, default=str))

    if args.tag:
        print(f"tagged run: outputs in {out}; models not saved")
        return

    # Final LightGBM on all wells (fixed tree count = mean best iteration).
    lab = df["Y"].notna().to_numpy()
    y = df.loc[lab, "Y"]
    final = lgb.LGBMClassifier(**LGB_PARAMS, random_state=args.seed,
                               scale_pos_weight=float((y == 0).sum() / y.sum()))
    final.fit(F[lab], y)
    final.booster_.save_model(str(ae.MODEL_DIR / "esp_failure_lgbm.txt"))
    # Alarm threshold = mean of the per-fold thresholds (each set on held-out wells
    # at the healthy-day quantile), so it reflects out-of-sample probabilities.
    META_PATH.write_text(json.dumps({
        "features": feats, "threshold": float(df["THR_LGB"].dropna().unique().mean()),
        "alarm_quantile": args.alarm_quantile, "persist_k": 3, "persist_n": 5,
        "horizon_days": HORIZON, "uses_ampcard": not args.no_ampcard,
        "cv": metrics["lightgbm"]}, indent=2, default=str))
    print(f"saved {ae.MODEL_DIR / 'esp_failure_lgbm.txt'}; outputs in {out}")


if __name__ == "__main__":
    main()
