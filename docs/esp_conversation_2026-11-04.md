# ESP session: risk plot, model parameters, current-only ablation (2026-11-04)

Follow-up to `esp_conversation_2026-11-02.md`. Results are in `esp_failure_risk_notes.md` section 7.

## You — 2026-11-04T08:08

Can you pull out the conversation we had about esp

## Claude — 2026-11-04T08:09

I'll look through your saved Claude Code sessions for this project.

## Claude — 2026-11-04T08:09

Almost every file matched because "esp" shows up inside words like "response". I'll search the actual user and assistant text for "ESP" as a whole word instead.

## Claude — 2026-11-04T08:09

One session matches: Nov 2, about ESP amp cards. I'll export it to readable markdown.

## Claude — 2026-11-04T08:09

I'll skim the rest so I can summarize it.

## Claude — 2026-11-04T08:09

I found it. Of your saved sessions, only one is about ESP: the Nov 2, 2026 session in this project, roughly 12:30–13:30 UTC.

I exported it as a readable transcript of your messages and my replies, without the tool calls:
`/tmp/claude-1000/-home-rianr-pypro-myvenv-esp-forecasting-src/6a6f406a-d436-4a65-a579-1741fe420fdd/scratchpad/esp_conversation.md`

That folder is temporary and gets cleared when this session ends, so tell me if you want the transcript copied somewhere permanent like `docs/`.

**What the conversation covered:**
1. **Amp card research:** You asked whether amp-card literature applies to `wells_data_final.csv`. The answer was not directly, because the data has one value per well per day (95 wells, 821 days), while amp-card patterns show up within minutes or hours. Other findings:
   - `CURRENT` sits at exactly 20 A in 22% of rows, which looks like a pump-off floor.
   - `AMPERAGE` is mostly a manual reading carried forward from earlier days, not SCADA data.
2. **Failure model:** We built an LSTM autoencoder (`src/esp_lstm_ae.py`) and a LightGBM model that predicts failure within 30 days (`src/esp_failure_clf.py`), and added both to the Streamlit app (`risk_app.py`).
3. **Amp-card classification:** `src/esp_ampcard.py` uses an autoencoder to group current patterns into shape families named after the amp-card literature. We added these classes to the app and tested them as LightGBM features. They gave no meaningful accuracy gain, and you can turn them off with `--no_ampcard`.
4. **Saved notes:** During that session I also wrote `docs/esp_failure_risk_notes.md` (318 lines). It is a structured write-up of all of the above: literature links, data findings, how the pipeline works, results, the run order and next steps.

**Run order** from that session:
```bash
python src/esp_lstm_ae.py
python src/esp_ampcard.py
python src/esp_failure_clf.py
streamlit run risk_app.py
```

The full raw transcript is at `~/.claude/projects/-home-rianr-pypro-myvenv-esp-forecasting-src/0838d459-6096-43d5-a9a2-f86057d95fdf.jsonl`. You can also reopen that session with `claude --resume`.

## You — 2026-11-04T08:09

copy it to docs

## Claude — 2026-11-04T08:10

I copied the transcript to `docs/esp_conversation_2026-11-02.md`. I added the date to the name so it sits clearly next to `esp_failure_risk_notes.md`. It's a new file and isn't committed to git; I can commit it if you want.

## You — 2026-11-04T08:10

Ok great, you can also read the docs

## Claude — 2026-11-04T08:10

I can't open the docs yet. Your settings block me from reading files outside `src/`, and `docs/` is one level up at `/home/rianr/pypro/myvenv/esp_forecasting/docs`.

To let me in, run:

```
/add-dir /home/rianr/pypro/myvenv/esp_forecasting/docs
```

Or add the whole project root (`/home/rianr/pypro/myvenv/esp_forecasting`) so I can also see `model/`, `output/` and `risk_app.py`. Once that's done, I'll read `esp_failure_risk_notes.md` and the transcript.

## You — 2026-11-04T08:12

ok, first thing first on risk_app.py, what is the first plot?

## Claude — 2026-11-04T08:12

I still can't open `risk_app.py`. It's in the project root (`/home/rianr/pypro/myvenv/esp_forecasting/risk_app.py`), and you only added `docs/`, so the root is still blocked.

Run this and I'll open the app and walk you through the first plot:

```
/add-dir /home/rianr/pypro/myvenv/esp_forecasting
```

That also covers `docs/`, `model/` and `output/`.

## You — 2026-11-04T08:13

Ok tell me what is the first plot

## Claude — 2026-11-04T08:13

The first chart in the app is the **Well detail** chart (`well_figure()`, `risk_app.py:179`). It appears after the four headline numbers and the fleet table, once you pick a well from the dropdown. It's one tall Plotly figure with five panels stacked on a shared date axis, and the top panel is the main plot.

**Panel 1: 30-day failure risk**
- **Blue line:** the LightGBM model's daily score for that well, meaning how likely it is to fail within the next 30 days. It's a ranking score rather than a true probability, because the model weighted the rare failure cases more heavily during training.
- **Dashed grey line:** the alarm threshold set by the sidebar slider. By default it's set so the alarm fires on about 4% of normal days.
- **Red shaded bands:** periods when the well was in alarm. An alarm needs the risk to stay above the threshold on at least 3 of the last 5 days, so a single spike doesn't trigger one.
- **Red vertical lines labelled "Failure":** recorded failures, shown when "Show recorded failures" is ticked. These lines run through all five panels.

How to read it: a good result is the risk line climbing into a red band in the weeks before a red Failure line. A red band with no failure after it is a false alarm.

**Panels below it, on the same dates:**
2. **Amp-card class:** one square per day, placed in a row for that day's current pattern (Normal, Overload – gradual rise, Underload – sudden drop, Gassy/erratic, Cycling, and so on). The number in each row label, like "(2.3×)", shows how much more often a failure follows within 30 days than for normal days.
3. **Motor current (A)**
4. **Intake pressure (psi)**
5. **Intake temperature (°F)**

Panels 3–5 leave gaps on pump-off days, including the 20 A floor values, so shutdowns don't show up as false drops.

Hovering anywhere shows the values from all five panels for that date. Under the chart, the "Selected day" section shows that day's amp card next to its prototype shape, plus a chart of which features pushed the risk up or down.

## You — 2026-11-04T08:14

What Plot 1 parameters?

## Claude — 2026-11-04T08:14

Plot 1's blue line is the output of the LightGBM model, which uses **91 input features** computed from each well's daily data. The saved settings are in `model/esp_failure_lgbm.json`.

### Inputs, in 6 groups

**1. Sensor deviations: 40 features** (`build_features()`, `src/esp_failure_clf.py:37`)

These come from 10 signals:
- **7 raw sensors:** CURRENT, FREQUENCY, PRESS_INT, PRESS_DESC, TEMP_INT, TEMP_MOT, VIBRATION
- **3 derived signals:**
  - `AMP_RATIO` = current ÷ the well's 30-day median current
  - `AMP_PER_HZ` = current ÷ frequency
  - `PUMP_DP` = discharge pressure − intake pressure

Each signal is converted to a **z-score against the well's own baseline**. The baseline is the 60 days that end 30 days ago, so recent changes don't get absorbed into it. For each signal there are 4 features:

| Feature | Meaning |
|---|---|
| `_z` | How far today is from the baseline |
| `_z_mean7` | 7-day average deviation |
| `_z_std7` | 7-day variability (erratic behaviour) |
| `_z_trend7` | This week's average minus last week's (drift) |

**2. Shutdown behaviour: 4 features.** Any day with current ≤ 20 A counts as "pump off".
- `off_frac7` and `off_frac30`: share of days off in the last 7 and 30 days
- `off_starts30`: number of shutdowns in the last 30 days, a daily stand-in for amp-chart "cycling"
- `days_since_off`

**3. Run life: 1 feature.** `run_age` is the number of days since the last failure, or since the start of the record.

**4. Well tests: 7 features.** BSW, GLR and BFPD, each with its change over 30 days, plus the 30-day change in PIP.

**5. LSTM autoencoder errors: 12 features.** `AE_SCORE` is the overall score for how unusual the last stretch looks. There are 11 per-signal errors (`ERR_CURRENT`, `ERR_VIBRATION`, …, `ERR_OFF`), which show which signal looks unusual.

**6. Amp-card features: 27 features.**
- Today's card class: 14 yes/no flags
- How many days in the last 30 fell into each warning class (cycling, trip, erratic, overload-gradual, underload, speed change): 6 features
- Unusual-shape day count and the card's reconstruction error: 2 features
- Likely-cause flags (pump-off, inside pump, gas, fluid, mechanical): 5 features

### Model settings
- **Target:** whether a failure happens 1–30 days after this day
- **LightGBM:** 150 trees, learning rate 0.02, 7 leaves per tree, at least 500 samples per leaf, `reg_lambda=10`, 70% of rows and 50% of features sampled per tree
- **Failure weighting:** failure cases are weighted up by the healthy-to-failure ratio. That's why the risk is a ranking score, not a true probability.

### Settings for the plot lines
- **Dashed threshold:** 0.63 by default. It's set at the 98th percentile of healthy-day scores on held-out wells, which ends up alarming on about 4% of healthy days.
- **Red alarm bands:** the score is above the threshold on at least 3 of the last 5 days.

### Cross-validated performance (5 folds, split by well)

| Metric | Value |
|---|---|
| AUC | 0.65 |
| PR-AUC | 0.13 (failure-window days are 5.5% of all days) |
| Failures caught with an alarm within 30 days | 23% |
| Median warning time | 22 days |
| False alarms per well per year | 0.91 |

In short, the model warns early when it does catch a failure, but it misses about three out of four failures. The per-day bar chart under the main plot shows which of these features pushed a given day's risk up.

## You — 2026-11-04T08:15

So the first plot in streamlit app is not just current?

## Claude — 2026-11-04T08:16

That's right, it isn't just current. Plot 1 is the risk score from a model that combines all the sensors, the well tests and run age. Current is only one of the inputs.

I grouped the model's saved feature importance (`output/failure_clf/feature_importance.csv`) to see how much each type of input contributes:

| Input group | Share of model importance |
|---|---|
| Other sensors: intake/discharge pressure, intake/motor temp, vibration, frequency | **43%** |
| Well tests: BFPD, BSW, GLR, PIP changes | **20%** |
| Current itself: `CURRENT_*`, `AMP_RATIO_*`, `AMP_PER_HZ_*` | **13%** |
| Run age (days since last failure) | **10%** |
| Signals derived from current: pump-off flags, shutdown counts, amp-card classes | **9%** |
| LSTM autoencoder overall score (mixes all signals) | **4%** |

Counting everything built from current, it accounts for only about **23%** of what drives the risk line. The single strongest feature is `run_age`, followed by BFPD, intake temperature and intake pressure.

In the app:
- **Plot 1 (risk)** is the multi-signal model.
- **Panel 2 (amp-card class)** and **panel 3 (motor current)** are the current-only views.

If you want a risk line that uses only current, which is closer to how an amp card is read, I can train a second LightGBM on just the current-derived features and show it next to the full model. On these numbers it would probably be weaker, but it would show how much current alone can tell you.

## You — 2026-11-04T08:18

Can you print/save the first plot so we can make sure talking about the same thing

## Claude — 2026-11-04T08:18

I'll render the same chart the app draws by calling the app's own functions, and save it as an image.

## Claude — 2026-11-04T08:18

`kaleido`, which Plotly needs to export PNGs, isn't installed. I'll save an interactive HTML version instead; it opens in any browser and looks exactly like the app chart.

## Claude — 2026-11-04T08:19

I saved the chart for the well the app selects by default (the first one in the dropdown, which has the highest current risk), so it matches what you see when the app opens:

`/home/rianr/pypro/myvenv/esp_forecasting/output/figures/plot1_well_detail_DEEPWATER-MECD2B.html`

Open it in a browser; it's interactive, with the same hover and zoom as the app. It loads the Plotly library from the web, so it needs an internet connection the first time. I made it by calling the app's own `score()`, `fleet_table()` and `well_figure()` functions, so it's the same chart, not a re-creation.

**What it shows for DEEPWATER-MECD2B:**
- **Top panel, 30-day failure risk:** the blue line ends at **0.74**, above the dashed threshold at **0.63**. The well is in 🔴 Alarm on the last date, 2021-05-25.
- **One red "Failure" line on 2019-03-09.** That's the only recorded failure for this well, and its run age is now 808 days.
- **14 alarm days** across the whole record, shaded in red.
- **Below that:** the amp-card class lanes (currently "Normal operation – steady at baseline"), then motor current, intake pressure and intake temperature.

So the risk is high even though the amp card reads normal, which fits the point that Plot 1 isn't driven by current alone. A very long run age is the top driver in the model.

If you'd rather have a static PNG, for a report for example, I can install `kaleido` into your venv with `pip install kaleido` and export one. Or I can save any other well you pick.

## You — 2026-11-04T08:19

install kaleido and save it as PNG

## Claude — 2026-11-04T08:19

Installed. Now I'll export the PNG (kaleido v1 needs a Chrome/Chromium browser; I'll check that it finds one).

## Claude — 2026-11-04T08:19

Let me look at it to make sure it rendered correctly.

## Claude — 2026-11-04T08:20

I installed `kaleido` and saved the PNG:

`/home/rianr/pypro/myvenv/esp_forecasting/output/figures/plot1_well_detail_DEEPWATER-MECD2B.png`

The first export cut off the amp-card lane names and the y-axis numbers, so I re-exported it with a wider left margin. I checked the new image and everything is readable. The interactive HTML version is in the same folder. I didn't change the app itself, only the export script.

**What the image shows (DEEPWATER-MECD2B):**
- **Top panel, 30-day failure risk:** the only recorded failure is at the far left (red line, March 2019). After that the risk mostly stays between 0.1 and 0.55.
- **Risk peaks:**
  - **Around April 2020:** the risk climbs to about 0.62, just under the threshold. It lines up with the intake pressure jump to about 2,000 psi (panel 4) and the amp-card shift to "Overload – gradual rise / steady above baseline" (panel 2).
  - **March and April 2021:** two short alarm periods (pink shading).
  - **End of record (May 2021):** the risk jumps to 0.74, which is the current alarm.
- **Amp-card panel:** the well is mostly "Normal – steady at baseline". It has short spells of overload, recent trips and restarts. The current class at the end is Normal.
- **Motor current:** steady at about 29 A, stepping up to about 33 A from April 2020, then easing to about 31–32 A.

So we're looking at the same chart: the top panel is the risk score, and the four panels below it are the evidence on the same timeline. The current alarm comes while current and the amp card both look normal, which matches what we found earlier: run age (about 800 days) and the other signals are pushing the risk up, not current.

## You — 2026-11-04T08:20

Open the image for me

## Claude — 2026-11-04T08:21

I sent the image to your default image viewer, but I can't confirm from here that a window actually opened. If nothing appeared, open it from your file manager or run:

```
! xdg-open /home/rianr/pypro/myvenv/esp_forecasting/output/figures/plot1_well_detail_DEEPWATER-MECD2B.png
```

## You — 2026-11-04T08:22

Ok, the first plot (risk) what is it or what are they?

## Claude — 2026-11-04T08:22

The top panel is the **30-day failure risk** for this well. It holds five things:

**1. Blue line: the risk score**
- One point per day.
- It answers: given everything the well's data shows today, how likely is a failure in the next 30 days?
- It runs from 0 to 1, and higher means more likely. Treat it as a ranking ("riskier than usual") rather than a true percentage: a value of 0.74 doesn't mean a 74% chance of failure. The model was trained with failures weighted up because they're rare, which pushes all the scores higher.
- It's calculated fresh each day from that day's data: sensors, well tests, run age and amp-card pattern.

**2. Dashed grey line at 0.63: the alarm threshold**
- This is the cut-off for "high enough to worry about".
- It was set so that on healthy wells the score goes above it on only about 2% of days.
- You can move it with the "Risk threshold" slider in the app's sidebar.

**3. Pink shaded bands: alarm periods**
- An alarm needs more than one high day: the score has to be above the threshold on at least 3 of the last 5 days, so a one-day spike doesn't count.
- On this well there are two short bands in March–April 2021.

**4. Red vertical line labelled "Failure": a recorded failure**
- This comes from the data's `FAILURE` column, not from the model.
- It's there so you can check the model: a good result is the blue line rising into a pink band in the weeks before a red line.
- This well has only one failure, in March 2019. It sits at the very start of the record, so there's no earlier risk to compare it with.

**5. Axes**
- **x-axis:** date. It's shared with all the panels below, so you can read straight down.
- **y-axis:** risk score from 0 to about 0.8.

**What it says about this well right now:** the blue line ends at 0.74, above the dashed line, so the well is in alarm. In plain terms, it looks more like wells shortly before failure than usual. The most likely reason is its long run (about 800 days since the last failure) combined with other signals; it isn't a current problem.

How far to trust it: in testing, an alarm came before only about 1 in 4 real failures, and there was about one false alarm per well per year. Treat it as a "look at this well" flag rather than a prediction.

## You — 2026-11-04T08:22

So is not a prediction I assume?

## Claude — 2026-11-04T08:23

It is a prediction, but a weak one, so don't treat it as a reliable forecast.

**Why it counts as a prediction:**
- Each day's score uses only data up to that day; it never looks ahead.
- The model was trained to answer one forward-looking question: will this well fail in the next 30 days? So the blue line is a forecast by construction.

**Why it isn't a reliable forecast:**

| Limitation | What it means |
|---|---|
| **Not a probability** | 0.74 doesn't mean a 74% chance. It only means riskier than most days. |
| **Misses most failures** | In testing, only **23%** of failures had an alarm in the 30 days before. About 3 in 4 failures came without warning. |
| **False alarms** | About **0.9 per well per year**. Across 95 wells, that's around 85 false alarms a year. |
| **Overall accuracy is modest** | AUC 0.65, where 0.5 is a coin flip and 1.0 is perfect. It's better than chance, but not by much. |
| **Good warning time when it does catch one** | Median warning is **22 days** before the failure. |

**How to use it, then:** as a **screening or ranking tool**, not a forecast. "Look at these wells first" is fair; "this well will fail in the next 30 days" is not.

**Why it's weak:** with one value per day, most of the short-lived signs of failure (minutes to hours) are lost. There are also only about 105 failures to learn from, and we still don't know what failure codes 1 and 2 mean.

To make it a stronger prediction, these would probably help most, in order:
1. Find out what failure types 1 and 2 are, and model them separately.
2. Get higher-frequency data (hourly or minute-level current).
3. Add nameplate amps for each well.
4. Calibrate the score so that 0.3 actually means about a 30% chance.

## You — 2026-11-04T08:24

Ok, what are the parameters you used to make this plot?

## Claude — 2026-11-04T08:24

These are the exact settings I used to make the PNG. They're the app's defaults, so the image matches what the app shows when it first opens.

**Data and model**

| Setting | Value |
|---|---|
| Data file | `data/wells_data_final.csv` (the app's default) |
| Risk model | `model/esp_failure_lgbm.txt` (LightGBM, 91 features, amp-card features on) |
| Model settings | `model/esp_failure_lgbm.json` |
| Amp-card model | `model/esp_ampcard_ae.pt` |
| Autoencoder scores | `output/lstm_ae/scores.csv` (from `esp_lstm_ae.py`) |

**Which well**

| Setting | Value |
|---|---|
| Fleet filter | "Only wells reporting on the latest date" = on, latest date 2021-05-25 |
| Well | **DEEPWATER-MECD2B**: the first well in the dropdown, which has the highest current risk (0.74) |
| Date range | The whole record for that well |

**Risk panel**

| Setting | Value |
|---|---|
| Risk horizon | Failure within **30 days** |
| Alarm threshold (dashed line) | **0.63**, the app default: 0.6338 rounded to 2 decimals |
| How the threshold was set | 98th percentile of healthy-day scores on held-out wells |
| Alarm rule (pink bands) | Above the threshold on **≥ 3 of the last 5 days** |
| Show recorded failures | **On**: red line from `FAILURE`, marked on the first day of each failure |
| Pump-off rule | Days with current ≤ **20 A** count as off; panels 3–5 leave gaps on those days |

**Image export**

| Setting | Value |
|---|---|
| Function | `well_figure()` from `risk_app.py`, used unchanged |
| Size | 1400 px wide × 2 scale → **2800 × 2008 px** PNG |
| Changes for the export only | Added a title, widened the left margin to 330 px so labels aren't cut off, set a white background |
| Script | `/tmp/claude-1000/-home-rianr-pypro-myvenv-esp-forecasting-src/6a6f406a-d436-4a65-a579-1741fe420fdd/scratchpad/save_plot1.py` (temporary) |

The export script is in the temporary folder, so it will be deleted when this session ends. If you want to regenerate the plot, or export other wells or thresholds later, I can move it into `src/` as a small tool, for example `python src/export_well_plot.py --well DEEPWATER-UVG14O --threshold 0.5`.

## You — 2026-11-04T08:26

So what parameters you used to trained the model

## Claude — 2026-11-04T08:26

These are the training settings for the LightGBM risk model, taken from `src/esp_failure_clf.py`. The run settings are confirmed in `output/failure_clf/metrics.json`.

### 1. Target (what it learns to predict)

| Setting | Value |
|---|---|
| Failure event | `FAILURE` > 0, so codes 1 and 2 are treated the same |
| **Positive (Y = 1)** | Days **1–30 before** a failure |
| **Negative (Y = 0)** | "Healthy" days: more than 30 days before and more than 60 days after any failure |
| **Not used in training** | The failure day itself and the **60 days after** a failure, while the pump is being replaced and restarted |
| Class balance | About **5.5%** positive days |

### 2. Input preparation (same for every well)

| Setting | Value |
|---|---|
| Pump-off rule | Current ≤ **20 A** counts as off; sensor values on those days are blanked |
| Baseline for z-scores | Each well's own data from **60 days, ending 30 days ago** (at least 20 days needed) |
| Rolling windows | 7 days (mean, std, trend) and 30 days (shutdowns, well-test changes, amp-card counts) |
| Features | **91** (the 6 groups I listed earlier) |
| Leakage check | Every feature uses only past data |

### 3. LightGBM settings (`LGB_PARAMS`)

| Parameter | Value | Why |
|---|---|---|
| `objective` | binary | Fail within 30 days: yes or no |
| `n_estimators` | **150** trees, fixed | Early stopping on about 11 validation wells was too noisy |
| `learning_rate` | 0.02 | Slow, steady learning |
| `num_leaves` | **7** | Small trees to avoid overfitting |
| `min_child_samples` | **500** | Each leaf needs 500 days of data |
| `subsample` / `subsample_freq` | 0.7 / 1 | Each tree sees 70% of rows |
| `colsample_bytree` | 0.5 | Each tree sees 50% of features |
| `reg_lambda` | **10** | Strong L2 regularization |
| `scale_pos_weight` | healthy ÷ failure days (about 17) | Weights up the rare failure days; this is why the score isn't a true probability |
| `random_state` | 42 (+ fold number) | Reproducibility |

These are deliberately heavy regularization settings, because the data has only about 105 failures.

### 4. Validation

| Setting | Value |
|---|---|
| Method | **5-fold cross-validation grouped by well** (`GroupKFold`), so a well's data is never in both training and test |
| Validation wells | 15% of the training wells in each fold, used only to set the threshold |
| Threshold | **98th percentile** of healthy-day scores on validation wells (`--alarm_quantile 0.98`) |
| Final threshold | Average across the 5 folds = **0.634** |
| Alarm rule | Above the threshold on ≥ 3 of the last 5 days |

### 5. Final model
After cross-validation, the same settings are retrained on **all 95 wells**. The result is saved to `model/esp_failure_lgbm.txt`, and that's the model the app uses.

### Comparison models in the same run

| Model | AUC | Failures caught | False alarms per well-year |
|---|---|---|---|
| **LightGBM (used)** | **0.65** | **23%** | 0.91 |
| LSTM classifier (30-day sequences, 30 epochs) | 0.51 | 9% | 0.39 |
| LSTM autoencoder (anomaly score only) | 0.55 | 10% | 0.38 |

LightGBM was clearly the best of the three, so it's the one used in the app.

**To retrain with different settings:**
```bash
python src/esp_failure_clf.py --alarm_quantile 0.95   # lower threshold: more alarms, catches more failures
python src/esp_failure_clf.py --no_ampcard            # drop the amp-card features
python src/esp_failure_clf.py --tag test1             # experiment run; doesn't overwrite the saved model
```

One small inconsistency in the code: a comment above the final model says the tree count is the "mean best iteration", but it's actually fixed at 150. The behaviour is fine; only the comment is out of date. I can fix it if you like.

## You — 2026-11-04T08:28

So basically you predicted the failure based on current?

## Claude — 2026-11-04T08:28

No, the model doesn't predict failure from current alone. Current is one input among many, and not the most important one.

Here's how much each type of input contributes to the model's decisions:

| Input | Share | Examples |
|---|---|---|
| Other sensors | **43%** | Intake temp, intake pressure, motor temp, vibration, discharge pressure |
| Well tests | **20%** | BFPD, BSW, GLR, PIP changes |
| **Current and anything built from it** | **23%** | Current, amp ratio, amps per Hz, pump-off and shutdown counts, amp-card classes |
| Run age | **10%** | Days since last failure (the single strongest feature) |
| LSTM autoencoder score | **4%** | Mixes all signals |

So about three quarters of the prediction comes from things other than current.

**Why it might look current-based:**
- The project started from the amp-card question, which is about current only.
- The app shows the amp-card and motor-current panels right under the risk plot.
- Current is used to define pump-off days (≤ 20 A), and that rule affects how the other sensors are cleaned.

**Why current carries less weight:** with one reading per day, most current-based warning signs (spikes, cycling, gas locking) are averaged away. Pressure, temperature and well-test trends survive daily sampling better.

If you want a model based on current only, to see what amp-card-style analysis alone can do with this data, I can train one with just the current-derived features and compare it against the full model. That would tell you directly how much predictive power current has on its own.

## You — 2026-11-04T08:29

train the current-only model and compare

## Claude — 2026-11-04T08:29

I'll add a feature-set option to the training script and run it as tagged experiments, so the saved model the app uses isn't overwritten.

## Claude — 2026-11-04T08:29

Both runs have started in the background. I'll compare the results when they finish.

## Claude — 2026-11-04T08:29

That failure came from my wait script, not the training: its check exited the whole shell on the first pass instead of retrying. I'll look at the logs directly.

## Claude — 2026-11-04T08:29

Both runs are still going; nothing has been written to the logs yet. I'll wait on them properly this time.

## Claude — 2026-11-04T08:30

The training runs are still going, and I'll get a notice when they finish.

## You — 2026-11-04T08:44

how long will it take?

## Claude — 2026-11-04T08:44

The training already finished at 16:29. My wait script never stopped because it was detecting its own command line as a running job. I'll stop it and read the results.

## Claude — 2026-11-04T08:44

The training is already done: both runs finished within a few minutes of starting. My background wait script was the problem, and the "failed" notice you may see is just me stopping it; neither training run failed.

The models and the app weren't touched. The two experiments wrote only to `output/failure_clf_current/` and `output/failure_clf_current_runage/`.

## Results (5-fold cross-validation by well; 105 failures)

| Model | Features | AUC | PR-AUC | Failures caught ≤30 d | Median warning | False alarms per well-year |
|---|---|---|---|---|---|---|
| **Full model (current app)** | 91 | **0.65** | **0.13** | **23%** | 22 d | 0.91 |
| Current + run age | 35 | 0.63 | 0.09 | 15% | 27.5 d | 0.75 |
| **Current only** | 34 | 0.59 | 0.07 | 12% | 24 d | 1.11 |
| *Simple rule: current 20% off baseline* | 1 | 0.57 | 0.08 | 15% | 20.5 d | 0.51 |
| *Random guess* | – | 0.50 | 0.055 | – | – | – |

## What this means

1. **Current alone has very little predictive power.** An AUC of 0.59 is only a little better than a coin flip (0.50). Its PR-AUC of 0.07 is barely above the 0.055 you'd get by guessing.
2. **The machine-learning model on current isn't better than a simple rule.** "Alarm when current is 20% away from its baseline" catches as many failures (15% vs 12%) with half the false alarms. With daily data, a 34-feature model has no extra current pattern to find.
3. **Run age adds the most of anything here.** Adding it to the current features raises AUC from 0.59 to 0.63, and it becomes the top feature in that model (26% of importance).
4. **The other sensors and well tests carry most of the remaining signal.** Only the full model reaches PR-AUC 0.13 and catches nearly twice as many failures as the current-only model.

**Top current-only features:** 7-day average current deviation (15%), days since the last shutdown (13%) and how unusual the amp-card shape is (13%).

**One caveat:** the "failures caught" and "false alarms" columns don't use exactly the same alarm rate for every model. The full model alarms on 4.0% of healthy days and the current-only model on 2.4%, so part of the gap in failures caught comes from that. AUC and PR-AUC don't depend on a threshold, and they tell the same story.

**Bottom line for the amp-card question:** with daily data, current-based (amp-card-style) analysis can't predict ESP failures well. It's useful for describing what the pump is doing (normal, overload, cycling), but prediction needs the other sensors, the well tests and run age.

## Code change
I added two options to `src/esp_failure_clf.py` and haven't committed them:
- `--features {all,current,current_runage}`: pick a feature set. Anything other than `all` requires `--tag`, so it can't overwrite the app's model.
- `--no_lstm`: skip the LSTM comparison so the run is faster.

To re-run:
```bash
python src/esp_failure_clf.py --features current --no_lstm --tag current
```

I can add this comparison to `docs/esp_failure_risk_notes.md`, or commit the changes.

## You — 2026-11-04T08:45

add the comparison to the docs

## Claude — 2026-11-04T08:45

I added the comparison to `docs/esp_failure_risk_notes.md`.

**New subsection in section 7: "Is the risk model 'based on current'? Current-only ablation (2026-11-04)"**
- The breakdown of the full model's feature importance by input type, showing that current and everything built from it account for about 23%.
- Which features are in the `current` and `current_runage` sets, and why frequency-based and multi-sensor features were left out.
- The results table: full model, current + run age, current only, the ±20% amp-ratio rule, and random guessing.
- The four findings, the caveat that the models alarm at different rates, and the conclusion for the amp-card question.
- The commands to re-run it and where the outputs are.

**Section 6, Tuning:** added the new `--features` and `--no_lstm` options to the table.

**Section 8, Next steps:** marked the current-only test as done. I also added one item that came up earlier: calibrate the risk score so it reads as a real probability.

Nothing is committed yet. These are uncommitted:
- `docs/esp_failure_risk_notes.md` (updated)
- `docs/esp_conversation_2026-11-02.md` (new)
- `src/esp_failure_clf.py` (the two new options)
- `output/figures/` (the plot PNG and HTML)
- the two new result folders

Should I commit them?

## You — 2026-11-04T08:46

just save all the conversation

