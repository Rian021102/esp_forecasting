"""Amp-card pattern classification of daily ESP current with an autoencoder.

A "daily amp card" is a 30-day window of one well's motor current relative to
its own rolling baseline (60 days ending 30 days ago). Classes follow the classic
ammeter-chart literature, translated to what daily values can show:

Tier 1 - shutdown sequences (from pump-off timing; Cunningham & Riling Figs 5-6)
  Recent trip               ran, then shut down and is still off
                            (pumped off / undercurrent or underload trip)
  Restart after shutdown    off for >= 3 days, now running again
                            (restart attempt after a trip)
  Cycling                   >= 3 trips in 30 days (pump-off / gas-lock cycling)
  Extended shutdown         off > 80% of the card

Tier 2 - running shape (autoencoder on running days only; Cunningham & Riling
Figs 2-4, API RP 11S, Takacs 2009)
  Normal operation          smooth, steady load near baseline
  Gassy / erratic loading   rapid load variation (gas interference, slugging)
  Low amperage (underload)  load per Hz sustained/trending below baseline
  High amperage (overload)  load per Hz sustained/trending above baseline
  Speed change (VSD)        frequency moved > 5%: current follows the set-point
                            (no fixed-speed amp-chart equivalent; not a fault)
  Power fluctuations        NOT observable (needs voltage at seconds resolution)

Likely cause (literature pairing with intake pressure / well tests):
  underload + falling intake pressure  -> pumped off / inflow decline
  underload + stable intake pressure   -> inside the pump: wear, tubing leak, gas
  erratic / cycling + high GLR         -> gas interference / gas lock
  overload + rising BSW / vibration    -> heavier fluid / mechanical drag, solids

Autoencoder: 1-D conv AE on the running-current channel (off days masked from
the loss) -> 8-number shape code -> KMeans clusters -> each cluster named by the
majority tier-2 class of its cards. Cards it reconstructs badly are "unusual".

    python src/esp_ampcard.py
Outputs: output/ampcard/{cards.csv, clusters.csv, class_vs_failure.csv,
prototypes.png, summary.json}, model/esp_ampcard_ae.pt
"""
import argparse
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.cluster import KMeans

import esp_lstm_ae as ae

OUT = ae.ROOT / "output" / "ampcard"
MODEL_PATH = ae.MODEL_DIR / "esp_ampcard_ae.pt"
W = 30  # days per daily amp card

SHUTDOWN_CLASSES = ["Recent trip", "Restart after shutdown", "Cycling", "Extended shutdown"]
RUNNING_CLASSES = ["Normal operation", "Gassy / erratic loading", "Low amperage (underload)",
                   "High amperage (overload)", "Speed change (VSD)"]

TH = dict(extended_off=0.8, cycling_trips=3, restart_min_off=3, min_run_days=15,
          jitter=0.05, resid=0.08, low=0.85, high=1.15, trend=0.10, freq_change=0.05,
          pip_fall=-0.10, glr_high=1.5, bsw_rise=5.0, vib_rise=0.5)


# ------------------------------------------------------------------- the cards
def _rolling_base(s, wells):
    b = s.groupby(wells).transform(
        lambda x: x.shift(ae.BASE_LAG).rolling(ae.BASE_REF, min_periods=20).median())
    return b.groupby(wells).ffill()  # keep the baseline through long shutdowns


def build_cards(raw):
    df = raw.sort_values(["WELL_ID", "DATE"]).reset_index(drop=True).copy()
    wells = df["WELL_ID"]
    df["OFF"] = (df["CURRENT"] <= ae.CURRENT_FLOOR).astype(np.float32)
    run = df["OFF"] == 0
    cur = df["CURRENT"].where(run)
    freq = df["FREQUENCY"].where(run & (df["FREQUENCY"] > 25))
    df["BASE_AMPS"] = _rolling_base(cur, wells)
    df["BASE_HZ"] = _rolling_base(freq, wells)
    df["RATIO"] = (cur / df["BASE_AMPS"]).clip(0, 2)                  # NaN when off
    df["FRATIO"] = (freq / df["BASE_HZ"]).groupby(wells).ffill()
    df["RATIO_HZ"] = (df["RATIO"] / df["FRATIO"]).clip(0, 2)           # load per Hz
    pi = df["PRESS_INT"].where((df["PRESS_INT"] > 0) & run)
    df["PI_FF"] = pi.groupby(wells).ffill()
    glr_med = df.groupby("WELL_ID")["GLR(SCF/BFPD)"].transform("median")
    df["GLR_REL"] = df["GLR(SCF/BFPD)"] / glr_med.replace(0, np.nan)
    return df


def card_arrays(df):
    """Return cards (n, 4, W): [ratio, off, fratio, ratio_hz] and each card's end row."""
    X, ends = [], []
    cols = ["RATIO", "OFF", "FRATIO", "RATIO_HZ"]
    A = df[cols].to_numpy(np.float32)
    ok_base = df["BASE_AMPS"].notna().to_numpy()
    for _, idx in df.groupby("WELL_ID").indices.items():
        for t in range(W - 1, len(idx)):
            if not ok_base[idx[t - W + 1]]:
                continue
            X.append(A[idx[t - W + 1:t + 1]].T)
            ends.append(idx[t])
    return np.stack(X), np.array(ends)


# ------------------------------------------------------------ literature rules
def card_metrics(x):
    r, off, fr, rhz = x
    on = off == 0
    trips = int(((off[1:] == 1) & (off[:-1] == 0)).sum())
    # length of the off-run that ended most recently (if running now)
    last_off_run = 0
    if on[-1] and (~on).any():
        end = np.flatnonzero(~on)[-1]
        start = end
        while start > 0 and not on[start - 1]:
            start -= 1
        last_off_run = end - start + 1
    m = {"off_share": float(off.mean()), "trips": trips, "off_now": bool(off[-1]),
         "ran_earlier": bool(on[:-1].any()), "last_off_run": int(last_off_run),
         "run_days": int(on.sum())}
    if on.sum() < 10:
        return {**m, **dict.fromkeys(["level_hz", "trend_hz", "jitter", "resid", "freq_dev"], np.nan)}
    t = np.flatnonzero(on)
    v, vh = r[on], rhz[on]
    if not np.isfinite(vh).all():
        vh = np.where(np.isfinite(vh), vh, v)
    coef = np.polyfit(t, vh, 1)
    m.update(level_hz=float(np.nanmedian(vh[-10:])), trend_hz=float(coef[0] * W),
             jitter=float(np.median(np.abs(np.diff(v)))),
             resid=float(np.std(v - np.polyval(np.polyfit(t, v, 1), t))),
             freq_dev=_sustained_freq_dev(fr[on]))
    return m


def _sustained_freq_dev(f):
    """Set-point move: |median of last 10 running days - median of first 10| or vs baseline."""
    f = f[np.isfinite(f)]
    if len(f) < 10:
        return 0.0
    first, last = np.median(f[:10]), np.median(f[-10:])
    return float(max(abs(last - first), abs(last - 1)))


def shutdown_class(m):
    if m["off_share"] > TH["extended_off"]:
        return "Extended shutdown"
    if m["trips"] >= TH["cycling_trips"]:
        return "Cycling"
    if m["off_now"] and m["ran_earlier"]:
        return "Recent trip"
    if m["last_off_run"] >= TH["restart_min_off"]:
        return "Restart after shutdown"
    return None


def running_class(m):
    if m["run_days"] < TH["min_run_days"] or np.isnan(m["level_hz"]):
        return None
    if m["jitter"] > TH["jitter"] or m["resid"] > TH["resid"]:
        return "Gassy / erratic loading"
    # Load per Hz: a set-point change alone does not make under/overload.
    if m["level_hz"] < TH["low"] or m["trend_hz"] < -TH["trend"]:
        return "Low amperage (underload)"
    if m["level_hz"] > TH["high"] or m["trend_hz"] > TH["trend"]:
        return "High amperage (overload)"
    if m["freq_dev"] > TH["freq_change"]:
        return "Speed change (VSD)"   # current followed the set-point, load per Hz normal
    return "Normal operation"


def likely_cause(cls, w):
    """Pair the current pattern with intake pressure / well tests (literature)."""
    pi = w["PI_FF"].dropna()
    pi_chg = (pi.iloc[-5:].mean() / pi.iloc[:5].mean() - 1) if len(pi) >= 10 else np.nan
    glr = w["GLR_REL"].iloc[-1]
    bsw_chg = w["BSW"].iloc[-1] - w["BSW"].iloc[0]
    vib = w["VIBRATION"].dropna()
    vib_chg = (vib.iloc[-5:].mean() - vib.iloc[:5].mean()) if len(vib) >= 10 else np.nan
    if cls == "Low amperage (underload)":
        if pi_chg < TH["pip_fall"]:
            return "Pumped off / inflow decline (intake pressure falling)"
        if not np.isnan(pi_chg):
            return "Inside the pump: wear, tubing leak or gas (intake pressure stable)"
    if cls in ("Gassy / erratic loading", "Cycling", "Recent trip"):
        if glr > TH["glr_high"]:
            return "Gas interference / gas lock (high GLR)"
        if pi_chg < TH["pip_fall"]:
            return "Pump-off (intake pressure falling)"
    if cls == "High amperage (overload)":
        if bsw_chg > TH["bsw_rise"]:
            return "Heavier fluid / emulsion (water cut rising)"
        if vib_chg > TH["vib_rise"]:
            return "Mechanical drag / solids (vibration rising)"
    return ""


def describe_prototype(proto, med_jitter):
    """Name a cluster from its median running-current card -> (shape, literature class)."""
    sm = np.convolve(proto, np.ones(3) / 3, mode="valid")
    start, end = np.median(proto[:5]), np.median(proto[-5:])
    change = end - start
    biggest_step = np.max(np.abs(np.diff(sm))) * 3 if len(sm) > 1 else 0
    sudden = abs(change) > 0 and biggest_step > 0.5 * abs(change)
    if med_jitter > TH["jitter"]:
        return "Erratic", "Gassy / erratic loading"
    if abs(change) < 0.05:
        if end > 1 + 0.08:
            return "Steady above baseline", "High amperage (overload)"
        if end < 1 - 0.08:
            return "Steady below baseline", "Low amperage (underload)"
        return "Steady at baseline", "Normal operation"
    if change > 0:
        return ("Sudden rise" if sudden else "Gradual rise"), "High amperage (overload)"
    if abs(end - 1) < 0.08:
        return "Recovering to baseline", "Normal operation"
    return ("Sudden drop" if sudden else "Gradual decline"), "Low amperage (underload)"


# ------------------------------------------------------------------ autoencoder
class CardAE(nn.Module):
    """1-D conv autoencoder for the running-current shape (1 x 30) -> 8 numbers."""

    def __init__(self, latent=8):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(2, 16, 5, padding=2), nn.ReLU(),                 # ch: ratio, run-mask
            nn.Conv1d(16, 32, 5, stride=2, padding=2), nn.ReLU(),      # 30 -> 15
            nn.Conv1d(32, 32, 3, stride=2, padding=1), nn.ReLU(),      # 15 -> 8
            nn.Flatten(), nn.Linear(32 * 8, latent))
        self.dec = nn.Sequential(
            nn.Linear(latent, 32 * 8), nn.ReLU(), nn.Unflatten(1, (32, 8)),
            nn.ConvTranspose1d(32, 32, 3, stride=2, padding=1), nn.ReLU(),
            nn.ConvTranspose1d(32, 16, 5, stride=2, padding=2, output_padding=1), nn.ReLU(),
            nn.Conv1d(16, 1, 5, padding=2))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z)[:, 0], z


def ae_input(X):
    """Running-current channel (off days filled with 1 = baseline) + run mask."""
    r = np.nan_to_num(X[:, 0], nan=1.0)
    run = (X[:, 1] == 0).astype(np.float32)
    r = np.where(run == 1, r, 1.0).astype(np.float32)
    return np.stack([r, run], 1)


def train_ae(Xin, epochs, device, seed):
    torch.manual_seed(seed)
    model = CardAE().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    dl = torch.utils.data.DataLoader(torch.from_numpy(Xin), batch_size=512, shuffle=True)
    for _ in range(epochs):
        for xb in dl:
            xb = xb.to(device)
            xh, _ = model(xb)
            m = xb[:, 1]
            loss = (((xh - xb[:, 0]) ** 2) * m).sum() / m.sum()
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def encode(model, Xin, device):
    model.eval()
    zs, errs = [], []
    for i in range(0, len(Xin), 8192):
        xb = torch.from_numpy(Xin[i:i + 8192]).to(device)
        xh, z = model(xb)
        m = xb[:, 1]
        zs.append(z.cpu().numpy())
        errs.append(((((xh - xb[:, 0]) ** 2) * m).sum(1) / m.sum(1).clamp(min=1)).cpu().numpy())
    return np.concatenate(zs), np.concatenate(errs)


# ------------------------------------------------------------------- reporting
def failure_table(cards, col, pre, lab, base_rate):
    rows = []
    for name, sub in cards.groupby(col):
        sl = lab.loc[sub.index]
        fr = pre.loc[sub.index][sl].mean() if sl.any() else np.nan
        rows.append({col: name, "share_pct": round(100 * len(sub) / len(cards), 1),
                     "failure_within_30d_pct": round(100 * fr, 1),
                     "failure_lift": round(fr / base_rate, 2)})
    return pd.DataFrame(rows).sort_values("failure_lift", ascending=False)


def prototypes_png(Xin, cards, clusters, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n, cols = len(clusters), 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(14, 2.6 * rows), sharex=True, sharey=True)
    days = np.arange(-W + 1, 1)
    run_idx = cards.index[cards["CLUSTER"] >= 0]
    for ax, (_, c) in zip(np.atleast_1d(axes).flat, clusters.iterrows()):
        idx = np.flatnonzero(cards.loc[run_idx, "CLUSTER"].to_numpy() == c["cluster"])
        rng = np.random.default_rng(0)
        for s in Xin[rng.choice(idx, min(40, len(idx)), replace=False), 0]:
            ax.plot(days, s, color="#2a78d6", alpha=0.08, lw=1)
        ax.plot(days, np.median(Xin[idx, 0], axis=0), color="#2a78d6", lw=2)
        ax.axhline(1, color="#898781", lw=1, ls="--")
        ax.set_title(f"C{c['cluster']}: {c['shape']} → {c['amp_card_class']}\n"
                     f"{c['share_pct']}% of running cards · fail lift {c['failure_lift']:.1f}×",
                     fontsize=8.5)
        ax.set_ylim(0.4, 1.8)
    for ax in np.atleast_1d(axes).flat[n:]:
        ax.axis("off")
    fig.supxlabel("day of card (0 = today)")
    fig.supylabel("current / baseline (running days)")
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


# ------------------------------------------------------------ final classes
def finalize(cards, df, ends, cluster_class, cluster_shape, thr_unusual):
    """Combine tier-1 rules, AE cluster names and rule overrides (in place)."""
    cards["AE_CLASS"] = cards["CLUSTER"].map(cluster_class)
    cards["AE_SHAPE"] = cards["CLUSTER"].map(cluster_shape)
    cards["UNUSUAL"] = cards["AE_REC_ERR"] > thr_unusual
    # Shutdown sequence (rules) first; else the autoencoder cluster's literature class.
    # Erratic loading is defined by day-to-day jitter, which the autoencoder smooths
    # away by design, and a set-point move with normal load/Hz is a speed change,
    # so the rules keep those two classes.
    rule = cards["RUNNING_RULE_CLASS"]
    keep_rule = rule.isin(["Speed change (VSD)", "Gassy / erratic loading"])
    ae_cls = cards["AE_CLASS"].where(~keep_rule, rule)
    cards["AMP_CARD_CLASS"] = cards["SHUTDOWN_CLASS"].fillna(ae_cls) \
        .fillna(rule).fillna("Insufficient running data")
    from_ae = ~keep_rule & cards["SHUTDOWN_CLASS"].isna() & cards["AE_SHAPE"].notna()
    cards["AMP_CARD_DETAIL"] = cards["AMP_CARD_CLASS"].where(
        ~from_ae, cards["AMP_CARD_CLASS"] + " – " + cards["AE_SHAPE"].str.lower())
    cards["LIKELY_CAUSE"] = [likely_cause(c, df.iloc[e - W + 1:e + 1])
                             for c, e in zip(cards["AMP_CARD_CLASS"], ends)]
    return cards


def _rule_cards(df):
    X, ends = card_arrays(df)
    mets = [card_metrics(x) for x in X]
    cards = df.loc[ends, ["DATE", "WELL_ID"]].copy()
    cards["SHUTDOWN_CLASS"] = [shutdown_class(m) for m in mets]
    cards["RUNNING_RULE_CLASS"] = [running_class(m) for m in mets]
    for k in ["off_share", "trips", "level_hz", "trend_hz", "jitter", "resid", "freq_dev"]:
        cards[k] = [m[k] for m in mets]
    return X, ends, cards


def classify(raw, ckpt_path=MODEL_PATH, device=None):
    """Amp-card class per well-day with the saved autoencoder (inference).

    `raw` uses the wells_data_final.csv schema. Returns (cards, X, ckpt): cards is
    indexed like build_cards(raw) (sorted by WELL_ID, DATE); days without a full
    30-day card with baseline are absent. X holds the cards' channels
    [ratio, off, fratio, ratio_hz].
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    raw = raw.copy()
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    df = build_cards(raw)
    X, ends, cards = _rule_cards(df)
    running = (cards["SHUTDOWN_CLASS"].isna() & cards["RUNNING_RULE_CLASS"].notna()).to_numpy()
    model = CardAE().to(device)
    model.load_state_dict(ck["state_dict"])
    cards["CLUSTER"] = -1
    cards["AE_REC_ERR"] = np.nan
    if running.any():
        Z, rec = encode(model, ae_input(X[running]), device)
        d = ((Z[:, None, :] - ck["centroids"][None]) ** 2).sum(-1)
        cards.loc[running, "CLUSTER"] = d.argmin(1)
        cards.loc[running, "AE_REC_ERR"] = rec
    finalize(cards, df, ends, ck["cluster_class"], ck["cluster_shape"], ck["unusual_threshold"])
    return cards, X, ck


# ------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT.mkdir(parents=True, exist_ok=True)

    df = ae.mark_exclusion(build_cards(ae.load()))
    X, ends, cards = _rule_cards(df)
    cards = pd.concat([df.loc[ends, ["FAILURE", "T_TO_FAIL", "EXCLUDED"]], cards], axis=1)
    print(f"{len(cards)} daily amp cards, {df['WELL_ID'].nunique()} wells")

    # Autoencoder on running-shape cards (no shutdown sequence, enough run days).
    running = cards["SHUTDOWN_CLASS"].isna() & cards["RUNNING_RULE_CLASS"].notna()
    Xin = ae_input(X[running.to_numpy()])
    print(f"training running-shape autoencoder on {len(Xin)} cards")
    model = train_ae(Xin, args.epochs, device, args.seed)
    Z, rec = encode(model, Xin, device)
    km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(Z)
    cards["CLUSTER"] = -1
    cards.loc[running, "CLUSTER"] = km.labels_
    cards["AE_REC_ERR"] = np.nan
    cards.loc[running, "AE_REC_ERR"] = rec

    pre = (cards["T_TO_FAIL"] >= 1) & (cards["T_TO_FAIL"] <= ae.PRE_DAYS)
    lab = pre | ~cards["EXCLUDED"]
    base_rate = pre[lab].mean()

    rows = []
    run_cards = cards[running]
    for c, sub in run_cards.groupby("CLUSTER"):
        proto = np.median(Xin[km.labels_ == c, 0], axis=0)
        shape, lit = describe_prototype(proto, sub["jitter"].median())
        vc = sub["RUNNING_RULE_CLASS"].value_counts(normalize=True)
        sl = lab.loc[sub.index]
        fr = pre.loc[sub.index][sl].mean()
        rows.append({"cluster": int(c), "shape": shape, "amp_card_class": lit,
                     "purity": round(float(vc.get(lit, 0.0)), 3),
                     "rule_majority": vc.index[0],
                     "share_pct": round(100 * len(sub) / len(run_cards), 1),
                     "failure_within_30d_pct": round(100 * fr, 1),
                     "failure_lift": round(fr / base_rate, 2),
                     "median_level_hz": round(sub["level_hz"].median(), 2),
                     "median_jitter": round(sub["jitter"].median(), 3)})
    clusters = pd.DataFrame(rows).sort_values("failure_lift", ascending=False)
    name = clusters.set_index("cluster")["amp_card_class"]
    shape_name = clusters.set_index("cluster")["shape"]
    purity = clusters.set_index("cluster")["purity"]

    thr_unusual = float(np.quantile(rec, 0.99))
    finalize(cards, df, ends, name.to_dict(), shape_name.to_dict(), thr_unusual)
    cards["AE_CONFIDENCE"] = cards["CLUSTER"].map(purity)

    final_tab = failure_table(cards, "AMP_CARD_DETAIL", pre, lab, base_rate)
    rule_tab = failure_table(cards.assign(R=cards["SHUTDOWN_CLASS"].fillna(
        cards["RUNNING_RULE_CLASS"]).fillna("Insufficient running data")), "R", pre, lab, base_rate)
    shape_tab = failure_table(cards[running], "AE_SHAPE", pre, lab, base_rate)
    agree = (run_cards["RUNNING_RULE_CLASS"] == run_cards["CLUSTER"].map(name)).mean()  # vs rules
    unusual_tab = failure_table(cards[running].assign(
        U=cards.loc[running, "UNUSUAL"].map({True: "Unusual (top 1% AE error)",
                                             False: "Typical shape"})), "U", pre, lab, base_rate)

    print(f"\nbase rate failure within 30 d: {100 * base_rate:.1f}%")
    print("\nAMP-CARD CLASS (shutdown rules + autoencoder running shape) vs failure:")
    print(final_tab.to_string(index=False))
    print("\nfor comparison, rules only:")
    print(rule_tab.to_string(index=False))
    print("\nautoencoder shape families vs failure:")
    print(shape_tab.to_string(index=False))
    print("\nrunning-shape clusters (autoencoder), named by majority literature class:")
    print(clusters.to_string(index=False))
    print(f"\ncluster name agrees with each card's own literature class: {agree:.0%}")
    print("\nautoencoder 'unusual' shapes:\n", unusual_tab.to_string(index=False))
    print("\nlikely-cause annotations:\n",
          cards.loc[cards["LIKELY_CAUSE"] != "", "LIKELY_CAUSE"].value_counts().to_string())

    cards.to_csv(OUT / "cards.csv", index=False)
    clusters.to_csv(OUT / "clusters.csv", index=False)
    final_tab.to_csv(OUT / "class_vs_failure.csv", index=False)
    prototypes_png(Xin, cards, clusters, OUT / "prototypes.png")
    protos = {int(c): np.median(Xin[km.labels_ == c, 0], axis=0).tolist()
              for c in range(args.k)}
    lift = final_tab.set_index("AMP_CARD_DETAIL")["failure_lift"].to_dict()
    torch.save({"state_dict": model.state_dict(), "centroids": km.cluster_centers_,
                "cluster_class": name.to_dict(), "cluster_shape": shape_name.to_dict(),
                "cluster_purity": purity.to_dict(), "prototypes": protos,
                "class_failure_lift": lift, "base_rate": float(base_rate),
                "unusual_threshold": thr_unusual, "window": W, "thresholds": TH}, MODEL_PATH)
    (OUT / "summary.json").write_text(json.dumps(
        {"cards": len(cards), "running_cards": int(running.sum()), "k": args.k,
         "base_rate": base_rate, "cluster_rule_agreement": agree}, indent=2))
    print(f"\nsaved {MODEL_PATH}; outputs in {OUT}")


if __name__ == "__main__":
    main()
