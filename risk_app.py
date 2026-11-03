"""Streamlit app: 30-day ESP failure risk per well (LightGBM, daily data).

    streamlit run risk_app.py

Needs the trained models in model/ (run src/esp_lstm_ae.py then
src/esp_failure_clf.py). Input CSV uses the wells_data_final.csv schema;
the FAILURE column is optional.
"""
import io
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import esp_ampcard as ac  # noqa: E402
import esp_failure_clf as fc  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "wells_data_final.csv"
REQUIRED = ["DATE", "WELL_ID", "CURRENT", "FREQUENCY", "PRESS_INT", "PRESS_DESC",
            "TEMP_INT", "TEMP_MOT", "VIBRATION", "BSW", "GLR(SCF/BFPD)", "BFPD", "PIP(PSI)"]

# Reference data-viz palette (works on Streamlit light and dark themes).
BLUE, ORANGE, MUTED, CRITICAL = "#2a78d6", "#eb6834", "#898781", "#d03b3b"

# Amp-card lanes, top to bottom (most failure-associated first).
CARD_LANES = ["Cycling", "Recent trip", "Gassy / erratic loading",
              "High amperage (overload) – gradual rise", "High amperage (overload) – sudden rise",
              "High amperage (overload) – steady above baseline",
              "Low amperage (underload) – gradual decline", "Low amperage (underload) – sudden drop",
              "Low amperage (underload) – steady below baseline",
              "Normal operation – steady at baseline", "Normal operation – recovering to baseline",
              "Speed change (VSD)", "Restart after shutdown", "Extended shutdown",
              "Insufficient running data"]

SENSOR_PANELS = [("CURRENT", "Motor current (A)"),
                 ("PRESS_INT", "Intake pressure (psi)"),
                 ("TEMP_INT", "Intake temperature (°F)")]

NAMES = {"CURRENT": "Current", "FREQUENCY": "Frequency", "PRESS_INT": "Intake pressure",
         "PRESS_DESC": "Discharge pressure", "TEMP_INT": "Intake temp",
         "TEMP_MOT": "Motor temp", "VIBRATION": "Vibration", "AMP_RATIO": "Amp ratio",
         "AMP_PER_HZ": "Amps per Hz", "PUMP_DP": "Pump ΔP", "OFF": "Pump-off flag"}


def label(feat):
    """Readable name for a model feature."""
    fixed = {"run_age": "Run age (days since last failure)",
             "off_frac7": "Days off, last 7", "off_frac30": "Days off, last 30",
             "off_starts30": "Shutdowns, last 30 days",
             "days_since_off": "Days since last shutdown",
             "AE_SCORE": "Autoencoder anomaly score",
             "pip_test_chg30": "Well-test PIP change (30 d)",
             "BFPD": "Well-test gross rate", "BSW": "Well-test water cut",
             "GLR(SCF/BFPD)": "Well-test GLR"}
    if feat in fixed:
        return fixed[feat]
    if feat == "card_rec_err":
        return "Amp-card shape error (autoencoder)"
    if feat == "card_unusual_days30":
        return "Unusual amp cards, last 30 days"
    if feat.startswith("cause_"):
        return "Likely cause: " + feat[6:].replace("_", " ")
    m = re.match(r"card_(.+?)(_days30)?$", feat)
    if m:
        name = m[1].replace("_", " ").replace("off", "shutdown")
        return (f"Amp card '{name}' days, last 30" if m[2] else f"Amp card today: {name}")
    m = re.match(r"(.+)_chg30$", feat)
    if m:
        return f"{fixed.get(m[1], m[1])} change (30 d)"
    m = re.match(r"ERR_(.+)$", feat)
    if m:
        return f"Autoencoder error: {NAMES.get(m[1], m[1])}"
    m = re.match(r"(.+)_z(?:_(mean7|std7|trend7))?$", feat)
    if m:
        suffix = {None: "vs baseline", "mean7": "vs baseline, 7-day mean",
                  "std7": "7-day fluctuation", "trend7": "7-day trend"}[m[2]]
        return f"{NAMES.get(m[1], m[1])} {suffix}"
    return feat


AMP_CARD_HELP = """
A **daily amp card** is 30 days of motor current divided by the well's normal level
(rolling baseline: the 60 days ending 30 days ago). Classes follow Cunningham & Riling,
*Submersible Electrical Pump Ammeter Chart Interpretation* (Centrilift, SWPSC), API RP 11S
and Takács (2009), adapted to daily data:

| Class | How it is detected |
|---|---|
| Cycling | ≥3 trips in 30 days (pump-off / gas-lock cycling) |
| Recent trip | Ran, then shut down and still off (undercurrent / underload trip) |
| Restart after shutdown | Off ≥3 days, now running again |
| Extended shutdown | Off >80% of the card |
| Gassy / erratic loading | Day-to-day jitter >5% or scatter >8% (rule) |
| High / low amperage | Autoencoder shape family with current per Hz above / below baseline |
| Speed change (VSD) | Frequency set-point moved >5% while current per Hz stayed normal |
| Normal operation | Autoencoder shape family steady at (or recovering to) baseline |

**Running shapes** come from a convolutional autoencoder trained on running days only;
its codes are clustered and each cluster is named from its median card. **Likely cause** pairs
the class with intake pressure, GLR, water cut and vibration. Power fluctuations and true gas
locking cannot be seen in daily data. Retrain with `python src/esp_ampcard.py`.
"""


# ------------------------------------------------------------------------ data
@st.cache_data(show_spinner="Scoring wells…")
def score(csv_bytes):
    raw = pd.read_csv(io.BytesIO(csv_bytes))
    missing = [c for c in REQUIRED if c not in raw.columns]
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    df, contrib, card_out = fc.score_wells(raw, explain=True, return_cards=True)
    # Unscaled sensor values for plotting (same ordering as score_wells).
    raw["DATE"] = pd.to_datetime(raw["DATE"])
    raw = raw.sort_values(["WELL_ID", "DATE"]).reset_index(drop=True)
    view = df[["DATE", "WELL_ID", "FAIL_EVENT", "OFF", "RISK"]].copy()
    for c, _ in SENSOR_PANELS:
        view[c] = raw[c].where(df["OFF"] == 0)
    run_age = fc._days_since(df["FAIL_EVENT"].astype(bool), df["WELL_ID"], start_at_zero=True)
    view["RUN_AGE"] = run_age.to_numpy()

    # Amp-card class per day (autoencoder + literature rules).
    cards, X, ck = card_out if card_out is not None else ac.classify(raw)
    cards = cards.assign(CARD_POS=np.arange(len(cards)))
    cols = ["AMP_CARD_CLASS", "AMP_CARD_DETAIL", "LIKELY_CAUSE", "UNUSUAL", "CLUSTER",
            "AE_SHAPE", "CARD_POS"]
    view = view.merge(cards[["WELL_ID", "DATE"] + cols], on=["WELL_ID", "DATE"], how="left")
    card_info = {"ratio": X[:, 0], "off": X[:, 1], "prototypes": ck["prototypes"],
                 "lift": ck["class_failure_lift"], "base_rate": ck["base_rate"]}
    return view, contrib, card_info


def fleet_table(view, thr):
    rows = []
    for w, s in view.groupby("WELL_ID"):
        last = s.iloc[-1]
        recent = s[s["DATE"] > last["DATE"] - pd.Timedelta(days=30)]
        status = ("⚫ Pump off" if last["OFF"] else
                  "🔴 Alarm" if last["ALARM"] else
                  "🟡 Elevated" if last["RISK"] > 0.85 * thr else "🟢 Normal")
        rows.append({"Well": w, "Status": status, "Risk": last["RISK"],
                     "Amp card": last["AMP_CARD_DETAIL"] if isinstance(
                         last["AMP_CARD_DETAIL"], str) else "–",
                     "Alarm days (30 d)": int(recent["ALARM"].sum()),
                     "Run age (d)": int(last["RUN_AGE"]),
                     "Last date": last["DATE"].date()})
    return pd.DataFrame(rows).sort_values("Risk", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------- charts
def alarm_spans(s):
    a = s["ALARM"].to_numpy(bool)
    d = s["DATE"].to_numpy()
    edges = np.flatnonzero(np.diff(np.r_[0, a.astype(int), 0]))
    return [(d[i], d[min(j, len(d) - 1)]) for i, j in zip(edges[::2], edges[1::2])]


SHORT = [("High amperage (overload)", "Overload"), ("Low amperage (underload)", "Underload"),
         ("Normal operation", "Normal"), ("Gassy / erratic loading", "Erratic (gassy)"),
         ("Restart after shutdown", "Restart after shutdown"),
         ("Insufficient running data", "Insufficient data")]


def lane_label(name, lift):
    short = name
    for a, b in SHORT:
        short = short.replace(a, b)
    x = lift.get(name)
    return f"{short} ({x:.1f}×)" if x is not None and np.isfinite(x) else short


def well_figure(s, thr, show_failures, lift):
    present = set(s["AMP_CARD_DETAIL"].dropna())
    lanes = [c for c in CARD_LANES if c in present] + sorted(present - set(CARD_LANES))
    n_lane = max(len(lanes), 1)
    fig = make_subplots(rows=2 + len(SENSOR_PANELS), cols=1, shared_xaxes=True,
                        vertical_spacing=0.045,
                        row_heights=[0.3, 0.05 + 0.03 * n_lane] + [0.15] * len(SENSOR_PANELS),
                        subplot_titles=[f"30-day failure risk · dashed = alarm threshold {thr:.2f}"
                                        " · shaded = alarm",
                                        "Amp-card class (× = how often a failure follows "
                                        "within 30 d vs normal)"]
                        + [t for _, t in SENSOR_PANELS])
    fig.add_trace(go.Scatter(x=s["DATE"], y=s["RISK"], name="Risk", mode="lines",
                             line=dict(color=BLUE, width=2),
                             hovertemplate="%{x|%Y-%m-%d}<br>Risk %{y:.2f}<extra></extra>"),
                  row=1, col=1)
    fig.add_hline(y=thr, line=dict(color=MUTED, width=1.5, dash="dash"), row=1, col=1)
    for x0, x1 in alarm_spans(s):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=CRITICAL, opacity=0.12, line_width=0,
                      row=1, col=1)
    lab = {c: lane_label(c, lift) for c in lanes}
    sc = s[s["AMP_CARD_DETAIL"].notna()]
    cause = sc["LIKELY_CAUSE"].fillna("").replace("", "–")
    fig.add_trace(go.Scatter(
        x=sc["DATE"], y=sc["AMP_CARD_DETAIL"].map(lab), mode="markers", name="Amp card",
        marker=dict(symbol="square", size=6, color=BLUE, line=dict(width=0)),
        customdata=np.c_[sc["AMP_CARD_DETAIL"], cause],
        hovertemplate="%{x|%Y-%m-%d}<br>%{customdata[0]}<br>Likely cause: %{customdata[1]}"
                      "<extra></extra>"), row=2, col=1)
    fig.update_yaxes(categoryorder="array", categoryarray=[lab[c] for c in lanes][::-1],
                     tickfont=dict(size=10), row=2, col=1)
    for i, (c, t) in enumerate(SENSOR_PANELS, start=3):
        fig.add_trace(go.Scatter(x=s["DATE"], y=s[c], name=t, mode="lines",
                                 line=dict(color=BLUE, width=1.5), connectgaps=False,
                                 hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f}<extra>" + t
                                 + "</extra>"), row=i, col=1)
    if show_failures:
        for d in s.loc[s["FAIL_EVENT"] == 1, "DATE"]:
            fig.add_vline(x=d, line=dict(color=CRITICAL, width=1.5))
            fig.add_annotation(x=d, y=1, yref="y domain", text="Failure", showarrow=False,
                               xanchor="left", yanchor="top", font=dict(size=11))
    fig.update_layout(height=860 + 18 * n_lane, showlegend=False, hovermode="x unified",
                      margin=dict(l=10, r=10, t=40, b=10))
    fig.update_yaxes(range=[0, max(s["RISK"].max(), thr) * 1.18], row=1, col=1)
    return fig


def card_figure(r, info):
    """The selected day's 30-day amp card vs its autoencoder prototype."""
    days = np.arange(-ac.W + 1, 1)
    fig = go.Figure()
    if pd.notna(r["CARD_POS"]):
        pos = int(r["CARD_POS"])
        ratio = info["ratio"][pos].astype(float)
        off = info["off"][pos] == 1
        ratio[off] = np.nan
        if pd.notna(r["CLUSTER"]) and int(r["CLUSTER"]) >= 0:
            proto = info["prototypes"][int(r["CLUSTER"])]
            fig.add_trace(go.Scatter(x=days, y=proto, mode="lines", name="Cluster prototype",
                                     line=dict(color=MUTED, width=2, dash="dash"),
                                     hovertemplate="day %{x}: %{y:.2f}<extra>prototype</extra>"))
        fig.add_trace(go.Scatter(x=days, y=ratio, mode="lines+markers", name="This card",
                                 line=dict(color=BLUE, width=2), marker=dict(size=5),
                                 connectgaps=False,
                                 hovertemplate="day %{x}: %{y:.2f}× baseline<extra></extra>"))
        for d in days[off]:
            fig.add_vrect(x0=d - 0.5, x1=d + 0.5, fillcolor=MUTED, opacity=0.18, line_width=0)
    fig.add_hline(y=1, line=dict(color=MUTED, width=1))
    vals = [v for t in fig.data for v in t.y if v is not None and np.isfinite(v)]
    lo = min([0.4] + [v - 0.1 for v in vals])
    hi = max([1.6] + [v + 0.1 for v in vals])
    fig.update_yaxes(range=[max(lo, 0), hi])
    fig.update_layout(height=380, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="day of card (0 = selected date) · grey = pump off",
                      yaxis_title="current / baseline",
                      legend=dict(orientation="h", y=1.08, x=0))
    return fig


def drivers_figure(contrib_row, n=8):
    top = contrib_row.reindex(contrib_row.abs().sort_values(ascending=False).index)[:n][::-1]
    colors = [ORANGE if v > 0 else BLUE for v in top]
    fig = go.Figure(go.Bar(
        x=top.values, y=[label(f) for f in top.index], orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        hovertemplate="%{y}<br>%{x:+.3f} log-odds<extra></extra>"))
    fig.add_vline(x=0, line=dict(color=MUTED, width=1))
    fig.update_layout(height=40 * n + 60, margin=dict(l=10, r=10, t=10, b=30),
                      xaxis_title="← lowers risk · raises risk →", bargap=0.35)
    return fig


# ------------------------------------------------------------------------- app
def main():
    st.set_page_config(page_title="ESP Failure Risk", layout="wide")
    st.title("ESP Failure Risk")
    st.caption("Daily risk that an ESP fails within the next 30 days. LightGBM on "
               "amp-chart-style rolling features from daily SCADA and well-test data.")

    try:
        meta = fc.load_meta()
    except FileNotFoundError:
        st.error("Model not found. Run `python src/esp_lstm_ae.py` and "
                 "`python src/esp_failure_clf.py` first.")
        st.stop()

    with st.sidebar:
        st.header("Data")
        up = st.file_uploader("Upload well CSV (wells_data_final.csv schema)", type="csv")
        if up is None and not DEFAULT_DATA.exists():
            st.info("Upload a CSV to begin.")
            st.stop()
        csv_bytes = up.getvalue() if up else DEFAULT_DATA.read_bytes()
        st.caption(f"Using: {up.name if up else DEFAULT_DATA.name}")

        st.header("Alarm")
        thr = st.slider("Risk threshold", 0.05, 0.95, float(round(meta["threshold"], 2)), 0.01,
                        help="Default is the cross-validated threshold (alarms on ~4% of "
                             "normal days). Alarm = above threshold on ≥3 of the last 5 days.")
        show_failures = st.checkbox("Show recorded failures", value=True)
        st.header("Fleet")
        active_only = st.checkbox("Only wells reporting on the latest date", value=True,
                                  help="Wells whose record ended earlier are hidden.")

    try:
        view, contrib, card_info = score(csv_bytes)
    except ValueError as e:
        st.error(str(e))
        st.stop()
    view["ALARM"] = fc.alarm(view["RISK"], view["WELL_ID"], thr, meta)
    table = fleet_table(view, thr)
    latest = view["DATE"].max().date()
    if active_only:
        table = table[table["Last date"] == latest].reset_index(drop=True)

    # Fleet overview
    cv = meta["cv"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Wells in alarm", f"{(table['Status'] == '🔴 Alarm').sum()} / {len(table)}",
              help=f"Status on each well's last reported day (latest in file: {latest}).")
    c2.metric("Elevated", int((table["Status"] == "🟡 Elevated").sum()))
    c3.metric("CV: failures caught ≤30 d", f"{cv['recall_30d']:.0%}",
              help="5-fold cross-validation by well at the default threshold.")
    c4.metric("CV: false alarms / well-year", f"{cv['false_alarms_per_well_year']:.2f}")

    st.subheader("Fleet")
    st.dataframe(table, hide_index=True, width="stretch", height=280,
                 column_config={"Risk": st.column_config.ProgressColumn(
                     "Risk", min_value=0.0, max_value=1.0, format="%.2f")})

    # Single well
    st.subheader("Well detail")
    well = st.selectbox("Well (sorted by current risk)", table["Well"])
    s = view[view["WELL_ID"] == well].reset_index()
    last = s.iloc[-1]
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Latest risk", f"{last['RISK']:.2f}", f"{last['RISK'] - thr:+.2f} vs threshold",
              delta_color="inverse")
    k2.metric("Status", table.loc[table["Well"] == well, "Status"].item())
    k3.metric("Run age", f"{int(last['RUN_AGE'])} d")
    k4.metric("Alarm days (whole record)", int(s["ALARM"].sum()))

    st.plotly_chart(well_figure(s, thr, show_failures, card_info["lift"]), width="stretch",
                    theme="streamlit")

    st.subheader("Selected day: amp card and risk drivers")
    dates = s["DATE"].dt.date.tolist()
    day = st.select_slider("Date", options=dates, value=dates[-1])
    r = s.loc[s["DATE"].dt.date == day].iloc[0]
    row_idx = r["index"]
    left, right = st.columns(2)
    with left:
        cls = r["AMP_CARD_DETAIL"] if isinstance(r["AMP_CARD_DETAIL"], str) else None
        if cls:
            x = card_info["lift"].get(cls)
            st.markdown(f"**{cls}**" + (f" · failures follow {x:.1f}× as often as normal"
                                        if x is not None and np.isfinite(x) else ""))
            if isinstance(r["LIKELY_CAUSE"], str) and r["LIKELY_CAUSE"]:
                st.markdown(f"Likely cause: {r['LIKELY_CAUSE']}")
            if pd.notna(r["UNUSUAL"]) and bool(r["UNUSUAL"]):
                st.markdown("⚠️ Unusual shape: top 1% autoencoder reconstruction error")
        else:
            st.markdown("No amp card for this date (needs 30 days with a baseline).")
        st.plotly_chart(card_figure(r, card_info), width="stretch", theme="streamlit")
    with right:
        st.markdown(f"Risk **{r['RISK']:.2f}**: each feature's contribution "
                    "(LightGBM, log-odds) to that day's score")
        st.plotly_chart(drivers_figure(contrib.loc[row_idx]), width="stretch",
                        theme="streamlit")
    with st.expander("How the amp-card classes are defined"):
        st.markdown(AMP_CARD_HELP)

    with st.expander("Data table for this well"):
        st.dataframe(s[["DATE", "RISK", "ALARM", "OFF", "RUN_AGE", "AMP_CARD_DETAIL",
                        "LIKELY_CAUSE"]
                       + [c for c, _ in SENSOR_PANELS]], hide_index=True,
                     width="stretch")

    st.caption("Caveats: risk is a ranking score, not a calibrated probability. Daily "
               "data hides intra-day amp-chart patterns; expect most failures to be "
               "missed at low false-alarm rates. See output/failure_clf/metrics.json.")


if __name__ == "__main__":
    main()
