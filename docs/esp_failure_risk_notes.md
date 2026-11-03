# ESP Amp Card Research & 30-Day Failure Risk Model — Session Notes

Session date: 2026-11-02. Covers research, data findings, LSTM autoencoder, 30-day LightGBM model, Streamlit app, amp-card classification and the amp-card feature ablation. Data: `data/wells_data_final.csv`.

---

## 1. Research question: can amp-card (ammeter chart) analysis be used on this data?

### Short answer
Not in the classic sense, but the amp-chart logic can be adapted. A traditional amp chart is
a continuous current recording (24-hour or 7-day circular chart), and diagnosis relies on the
shape of the trace within that window. This dataset has one value per well per day, so
intra-day signatures are lost. Trend-based amp-chart logic (drift, under/overload against a
baseline, shutdown frequency) still applies.

### Classic amp-chart patterns vs daily data

| Classic pattern | Time scale | Visible daily? |
|---|---|---|
| Normal (smooth, flat line) | Hours | Yes, as a stable baseline |
| Gas interference (fluctuating current) | Minutes | Only weakly, via day-to-day variance |
| Gas lock, pump-off, underload trip, repeated restarts | Minutes–hours | No (and the 20 A floor hides it) |
| Gradual amp decline (wear, falling inflow, pump-off trend) | Days–weeks | Yes |
| Gradual amp rise (scale, sand, emulsion, viscosity, mechanical drag) | Days–weeks | Yes |
| Power supply / voltage problems | Seconds | No (`OUT_VOLT` also ~53% missing) |

### Literature

Guidebooks and standards
- **API RP 11S**, *Operation, Maintenance and Troubleshooting of ESP Installations* — the standard
  reference for ammeter chart interpretation. https://standards.globalspec.com/std/988038/api-rp-11s
- **Takács, G. (2009)**, *Electrical Submersible Pumps Manual*, Gulf Professional Publishing —
  chapter on amp charts and troubleshooting.
- **Bates et al. (2004)**, "Taking the Pulse of Producing Wells—ESP Surveillance", *Oilfield Review* 16.
- Baker Hughes (Centrilift) *Submersible Pump Handbook* — classic catalogue of amp-chart patterns
  (no online copy found).

Papers on current-based diagnosis
- **Han et al. (2015)**, "Real-Time Monitoring and Diagnosis of ESP", SPE-174873-MS.
- **Diagnosis of Operating Conditions of the ESP via Machine Learning (2022)**, *Sensors* 23(1):279.
  https://pmc.ncbi.nlm.nih.gov/articles/PMC9823322/ — 24-hour current traces (~344 points/day) from
  20 Brazilian wells, 4 classes (normal, normal with gas, gas interference, gas lock), ~93% with SVM.
  Shows the resolution amp-chart classification needs.
- **Han et al. (2025)**, "Fault Diagnosis of ESP Based on Motor Current Signal Analysis and Deep
  Learning", *SPE Journal* 30(5).
  https://faculty.cup.edu.cn/hanguoqing/lwcg/be18b42756c443b1bc4efe6a53231637.html
- **Liu et al. (2024)**, hybrid LSTM-CNN for ESP condition prediction, *SPE Journal* 29(5).
- **SPE 229219**, physics-based rules + ML for ESP condition monitoring.
  https://www.citedrive.com/en/discovery/machine-learning-provides-reliable-esp-condition-monitoring/
- **Explainable-AI ESP failure diagnosis (2025)**, *J. Chem. Pet. Eng.* https://jchpe.ut.ac.ir/article_107377.html
- **KUNGFU.AI case study** — 24-hour averages still held enough signal to predict failure; intake
  pressure tends to rise before solids-related failures (not peer-reviewed).
  https://www.kungfu.ai/case-study/advisory-exploration-and-proof-of-concept-with-drilling-technology-leader

Only the *Sensors* paper was read in full; SPE items are from listings/summaries — verify on OnePetro.

---

## 2. Data findings

- 58,979 rows, 95 wells, 821 days, strictly daily steps.
- **`AMPERAGE` is not SCADA.** It changes on the same days as the well-test columns (`BFPD` 6×,
  `PIP(PSI)` 8×, `FREC(Hz)` 7× their usual change rate), is an integer 86% of the time, and changes
  only every ~4 days (median). It is a well-test amp reading carried forward.
- Column groups:
  - **Sensor (SCADA/gauge), daily:** `CURRENT`, `FREQUENCY`, `PRESS_INT`, `PRESS_DESC`, `TEMP_INT`,
    `TEMP_MOT`, `OUT_VOLT`, `VIBRATION`
  - **Well test, held between tests:** `AMPERAGE`, `BFPD`/`BOPD`/`BWPD`, `BSW`, `PIP(PSI)`,
    `FREC(Hz)`, `WHP(PSI)`, `GOR`, `GLR`, `API`
- `CURRENT` = 20 A exactly in 22% of rows → treated as **pump off**.
- Other floors treated as missing: `PRESS_INT`/`PRESS_DESC` = 0, temperatures ≤ 175, frequency ≤ 25.
- 105 failure events (`FAILURE` = 1: 85, = 2: 20); 85 wells with one failure, 10 with two.
  Any `FAILURE` > 0 is treated as a failure (meaning of 1 vs 2 unconfirmed).

---

## 3. Pipeline

```
wells_data_final.csv
  → src/esp_lstm_ae.py
       engineer()        clean, pump-off flag, AMP_RATIO / AMP_PER_HZ / PUMP_DP
       mark_exclusion()  [-30, +60] days around failures; T_TO_FAIL
       scale_per_well()  rolling baseline: 60 days ending 30 days ago (median / IQR)
       LSTM autoencoder  trained on healthy 30-day windows → anomaly score + per-variable error
  → src/esp_failure_clf.py
       build_features()  64 causal features
       LightGBM          "failure within 30 days?"
       threshold + ≥3-of-5-days persistence → alarm
  → risk_app.py (Streamlit)
```

### Key design choices
- **Per-well, rolling baseline** instead of nameplate amps (not in the data). A first version using
  the well's first 180 days scored AUC 0.49 (random) because pump replacements shift the baseline.
- **Engineered amp-chart features:** `AMP_RATIO` = current / 30-day median (pseudo under/overload),
  `AMP_PER_HZ` = current / frequency (load vs VSD set-point), `PUMP_DP` = discharge − intake pressure.
- **Autoencoder:** 2-layer LSTM encoder (64) → 16-d latent → 2-layer LSTM decoder; masked MSE;
  score = error over last 7 days of each 30-day window; threshold = 99th pct of healthy validation days.
- **Labels:** 1 on days 1–30 before a failure, 0 on healthy days, failure day and 60 days after unused.
- **LightGBM:** lr 0.02, 7 leaves, min 500 samples/leaf, subsample 0.7, colsample 0.5, λ 10,
  150 trees (fixed — early stopping on ~11 val wells stopped at 1–2 trees), `scale_pos_weight` for imbalance.
- **Evaluation:** 5-fold CV grouped by well (all 105 failures scored out-of-sample); threshold set
  on held-out validation wells inside each fold.

### Feature groups (64)
- Deviation from baseline for 10 continuous variables: today, 7-day mean, 7-day std, 7-day trend
- Shutdowns: off share 7/30 days, shutdown count 30 days (daily "cycling"), days since last off
- Run age: days since last failure
- Well test: `BSW`, `GLR`, `BFPD` and 30-day changes; 30-day change in test `PIP`
- Autoencoder anomaly score + 11 per-variable errors

---

## 4. Results (5-fold CV by well, 105 failures)

Day-level ranking:

| Model | AUC | PR-AUC (base 5.5%) |
|---|---|---|
| **LightGBM** | **0.654** | **0.123** |
| Rule: current ±20% from 30-day median | 0.571 | 0.078 |
| LSTM autoencoder | 0.554 | 0.077 |
| LSTM classifier | 0.513 | 0.070 |

At matched alarm budget (5% of healthy days):

| Model | Caught ≤30 d | Median warning | False alarms / well-year |
|---|---|---|---|
| **LightGBM** | **24%** | 24 d | **0.80** |
| Amp-ratio rule | 23% | 20 d | 1.27 |
| LSTM autoencoder | 19% | 23 d | 1.26 |
| LSTM classifier | 13% | 15 d | 0.79 |

Top drivers: run age; intake pressure drifting up (7-day mean); intake temperature level and
fluctuation; days since last shutdown; autoencoder anomaly score; well-test `BFPD`, `GLR`, `BSW`.
Removing run age and well-test features barely changes AUC (0.673 vs 0.676 in the diagnostic).

Detection by failure type (LightGBM): type 1 21% (n=85), type 2 35% (n=20).

### Caveats
- Daily resolution is the ceiling; higher-frequency current data would help more than a better model.
- Risk is a ranking score (class-weighted), not a calibrated probability.
- 150-tree setting chosen after an out-of-fold diagnostic (flat curve 50–400 trees → small optimism).
- ~±8 percentage points uncertainty on recall with 105 events.
- On training wells, app scores correlate 0.78 with the out-of-sample CV scores.

---

## 5. Files

| File | Purpose |
|---|---|
| `src/esp_lstm_ae.py` | Data prep, rolling baseline, LSTM autoencoder, rule baselines, evaluation helpers, `ae_errors()` |
| `src/esp_failure_clf.py` | Features, LightGBM + LSTM classifier, CV, saving, `score_wells()` for inference |
| `src/esp_ampcard.py` | Daily amp-card classes (shutdown rules + conv autoencoder running shapes), `classify()` |
| `risk_app.py` | Streamlit dashboard |
| `model/esp_lstm_ae.pt` | Final autoencoder |
| `model/esp_ampcard_ae.pt` | Amp-card autoencoder, cluster names, prototypes, class failure lifts |
| `model/esp_failure_lgbm.txt` | Final LightGBM (91 features incl. amp card) |
| `model/esp_failure_lgbm.json` | Feature list, threshold (0.634), persistence rule, `uses_ampcard`, CV metrics |
| `output/lstm_ae/` | `scores.csv`, `events.csv`, `metrics.json` |
| `output/failure_clf/` | `predictions.csv`, `events.csv`, `metrics.json`, `feature_importance.csv` |
| `output/ampcard/` | `cards.csv`, `clusters.csv`, `class_vs_failure.csv`, `prototypes.png` |
| `output/failure_clf_base/`, `output/failure_clf_ampcard/` | Ablation runs without / with amp-card features |

The older apps (`isoapp.py`, `amperedet.py`, `lstmanodet.py`) expect the earlier CSV schema
(`Well`, `Date`, `Ampere`) and were left untouched. `app.py` is a FastAPI service.

---

## 6. How to run

```bash
cd /home/rianr/pypro/myvenv/esp_forecasting
source ../bin/activate.fish        # bash/zsh: source ../bin/activate
streamlit run risk_app.py          # models already trained
```

Retrain (order matters):

```bash
python src/esp_lstm_ae.py          # 1. LSTM autoencoder (~1–2 min on GPU)
python src/esp_ampcard.py          # 2. amp-card autoencoder (~20 s)
python src/esp_failure_clf.py      # 3. 30-day model (uses 1 and 2) + saved metadata
streamlit run risk_app.py          # 4. app (first load ~17 s, then cached)
```

### App contents
- Sidebar: default data or CSV upload (same schema, `FAILURE` optional), threshold slider
  (default 0.63), show failures, "only wells reporting on the latest date" (default on).
- Fleet: wells in alarm / elevated, CV metrics, table ranked by latest risk with status
  (🔴 Alarm / 🟡 Elevated / 🟢 Normal / ⚫ Pump off).
- Well detail: risk chart with threshold, alarm shading, failure markers; panels for motor current,
  intake pressure, intake temperature.
- What drives the risk: per-day LightGBM feature contributions (log-odds).
- Amp card (added later, see section 7): "Amp card" column in the fleet table; an amp-card
  class lane panel under the risk chart (each lane labelled with its failure lift); and for the
  selected date, the 30-day card vs its autoencoder cluster prototype with class, likely cause
  and "unusual" flag, next to the risk drivers. Amp cards are computed once inside the risk
  scoring and reused; the drivers chart labels amp-card features (e.g. "Amp card 'recent trip'
  days, last 30").

### Tuning

| Goal | Where |
|---|---|
| Fewer/more alarms | App slider; `--alarm_quantile` in `esp_failure_clf.py` |
| Different horizon | `PRE_DAYS` in `esp_lstm_ae.py` |
| Pump-off definition | `CURRENT_FLOOR` |
| Baseline window | `BASE_LAG`, `BASE_REF` |
| Alarm rule | `persist_k` / `persist_n` in the JSON, or `persist()` |

---

## 7. Amp-card pattern classification with an autoencoder (`src/esp_ampcard.py`)

### Can `CURRENT` be read as SCADA?
Yes — it changes on 54% of days and 32% of values have more than 2 decimals (likely daily
averages of higher-frequency readings). But it is one value per day, so each "amp card" is a
**30-day window of daily values**, not a 24-hour trace.

A first rule-only prototype (scratchpad, not in repo) found cycling, erratic and step-change
windows about 1.7× more often before failures, but they covered only ~7% of days, and 49 of
105 failures still looked "normal" the day before.

### Literature classes used
Main source: Cunningham & Riling (Byron Jackson Centrilift), *Submersible Electrical Pump Ammeter
Chart Interpretation*, Southwestern Petroleum Short Course —
https://www.swpshortcourse.org/paper/submersible-electrical-pump-ammeter-chart-interpretation.
Classes: normal operation (Fig 2), power fluctuations (Fig 3), gas locking (Fig 4), pumped off and
shut down on undercurrent (Fig 5), shut down on underload then failed restart (Fig 6), gassy /
erratic operation, low and high amperage operation. Also API RP 11S, Takács (2009), and
practitioner guides for the intake-pressure pairing (falling amps + falling intake pressure →
pumped off; falling amps + stable intake pressure → problem inside the pump).

### How it works
1. **Card:** per well-day, 30 days of current ÷ rolling baseline (60 days ending 30 days ago) plus a
   pump-off channel → 47,291 cards.
2. **Tier 1 – shutdown sequences (rules on pump-off timing):** Recent trip, Restart after shutdown,
   Cycling (≥3 trips), Extended shutdown (>80% off).
3. **Tier 2 – running shape (autoencoder):** 1-D conv autoencoder trained on running days only
   (off days masked), 8-number code, KMeans k=10.
4. **Naming:** each cluster named from its median card's shape (steady at/above/below baseline,
   gradual/sudden rise, sudden drop, recovering) → literature class (normal / high / low amperage).
   Rules override for **erratic loading** (defined by jitter, which the AE smooths) and **speed
   change (VSD)** (frequency moved >5% with normal current per Hz).
5. **Likely cause** from intake pressure, GLR, BSW and vibration changes.
6. **Unusual:** cards in the top 1% of AE reconstruction error.

### Iterations
- v1: AE on current + off channel → clusters organised by *when the pump switched off*, not shape.
  Fixed by training on running days only.
- v2: "overload" clusters at a steady 1.4× baseline almost never failed → drive speed-ups.
  Level classes now use current per Hz; data-estimated current–frequency exponent was noisy
  (regression 0.3, median day ratio 1.1), so simple current ÷ frequency is used.
- v3: frequency rule used the single largest daily deviation → 45% "speed change". Fixed with a
  sustained (10-day median) set-point move.
- v4: clusters named by majority of rule labels mislabelled a sudden step as "erratic" → named from
  prototype shape instead; erratic kept from rules; shape kept in the label.

### Results (base rate of failure within 30 d: 5.6%)

| Amp-card class | Share of cards | Failure within 30 d | Times base |
|---|---|---|---|
| Cycling | 2.3% | 13.7% | 2.47× |
| Gassy / erratic loading | 6.0% | 8.6% | 1.54× |
| Recent trip | 4.7% | 7.9% | 1.42× |
| High amperage (overload) – gradual rise | 2.8% | 7.1% | 1.28× |
| Normal operation – steady at baseline | 50.0% | 5.7% | 1.02× |
| High amperage (overload) – steady above baseline | 7.1% | 4.9% | 0.88× |
| Extended shutdown | 12.7% | 4.4% | 0.79× |
| Restart after shutdown | 6.5% | 3.3% | 0.60× |
| Speed change (VSD) | 6.5% | 2.7% | 0.49× |
| Small clusters (sudden rise/drop, recovering) | ~1.3% | 0% | — (likely baseline artefacts) |

AE "unusual" cards (top 1% reconstruction error): 1.85× base.

### Limits
- Power fluctuations and true gas locking are not observable at daily resolution.
- Thresholds are an interpretation of the literature, not values from API RP 11S (full text not accessed).
- Signals are modest; useful as interpretable labels, not as a stand-alone warning.

### Files
- `src/esp_ampcard.py` — run `python src/esp_ampcard.py` (~20 s on GPU)
- `output/ampcard/prototypes.png` (median card per shape family), `cards.csv`, `clusters.csv`,
  `class_vs_failure.csv`, `summary.json`
- `model/esp_ampcard_ae.pt`

### Amp card as LightGBM features (ablation)
27 features added in `esp_failure_clf.py` (`ampcard_features()`): today's class (one-hot),
days in the last 30 in cycling / recent trip / erratic / overload gradual rise / underload /
speed change, unusual-card count, card AE reconstruction error, likely-cause flags.
Now the default (`uses_ampcard: true` in `model/esp_failure_lgbm.json`); switch off with
`python src/esp_failure_clf.py --no_ampcard`. `--tag <name>` writes a comparison run to
`output/failure_clf_<name>` without saving models.

| Alarm budget | Without amp card: caught / FA per well-year | With amp card |
|---|---|---|
| 1% of healthy days | 11.4% / 0.30 | 10.5% / 0.29 |
| 2% | 13.3% / 0.40 | 16.2% / 0.45 |
| 5% | 23.8% / 0.80 | 23.8% / 0.95 |

Well-bootstrap (300×): PR-AUC +0.007 [−0.008, +0.027], AUC −0.001 [−0.011, +0.009] — no
significant change. Amp-card features take 5% of model gain (top one ranks 24 of 91).
Reason: existing shutdown and current-deviation features already carry the same information.
Caveat: the card autoencoder is trained on all wells (unsupervised, no labels), not per fold.
Comparison runs: `output/failure_clf_base/`, `output/failure_clf_ampcard/`.

---

## 8. Open questions / next steps
- Confirm what `FAILURE` = 1 vs 2 means; consider a separate model per failure type.
- Obtain raw SCADA/VSD current at 1 s–15 min intervals for true amp-chart classification.
- Confirm whether `CURRENT` = 20 A means "off" or "missing".
- ~~Add the amp-card class to the app; test it as LightGBM features~~ — done; no significant
  gain (see section 7). Decide whether to keep it in the model or revert with `--no_ampcard`.
- Verify amp-card thresholds against API RP 11S / Takács and engineer-labelled examples.
