# ESP Amp Card conversation (2026-11-02)

## You — 2026-11-02T12:30

Ok I'm researching about Electric Submersible Pump Amp Card. In the Data folder you will find final_well.csv, where there is amperage data. Find paper, journal, guidebook about amp card and could it be used for this type of data?

## Claude — 2026-11-02T12:30

The file is `wells_data_final.csv` (no `final_well.csv`), so I'll use that.

## Claude — 2026-11-02T12:32

I couldn't find `final_well.csv`. The file in `data/` is `wells_data_final.csv`, so that's what I analysed.

## Short answer

**Not in the classic sense, but the amp-chart logic can be adapted to your data.** A traditional amp chart is a continuous recording of motor current, usually on a 24-hour or 7-day circular chart, and diagnosis relies on the shape of the trace within that window. Your data has **one value per well per day**: 95 wells, 821 days, strictly 1-day steps. Most amp-chart signatures happen within minutes or hours, so a daily value cannot show them.

## What I found in your data

| Check | Result | What it means |
|---|---|---|
| Sampling | Daily, 58,979 rows | The patterns inside a chart (spikes, oscillation, cycling) are lost |
| `CURRENT` at exactly 20 A | 22% of rows | Looks like a floor or clip value. Shutdowns (0 A) are probably hidden, and shutdowns are the most important amp-chart event |
| `AMPERAGE` unchanged from the day before | 91% of rows (`CURRENT`: 46%) | `AMPERAGE` looks like a manual or well-test reading carried forward, not a sensor |
| `CURRENT` vs `AMPERAGE` correlation | 0.89 | Related but not the same. Use `CURRENT` as the trend signal |
| `FAILURE` | 85 events labelled "1", 20 labelled "2", spread over all 95 wells | You'll need to find out what 1 and 2 mean |
| Nameplate amps | Not in the data | Classic underload/overload thresholds can't be applied directly |

As an example, well `000XAR` sits at 20 A for several days, then jumps to about 69 A shortly before its failure label. That looks like a shutdown or restart sequence, not a gradual amp-chart trend.

## How amp-chart patterns carry over to daily data

| Classic pattern | Time scale | Visible daily? |
|---|---|---|
| Normal (smooth, flat line) | Hours | Yes, as a stable baseline |
| Gas interference (fluctuating current) | Minutes | Only weakly, through day-to-day variance |
| Gas lock, pump-off, underload trip, repeated restarts | Minutes to hours | No, and the 20 A floor hides it further |
| Gradual amp decline (wear, falling inflow, pump-off trend) | Days to weeks | **Yes** |
| Gradual amp rise (scale, sand, emulsion, viscosity, mechanical drag) | Days to weeks | **Yes** |
| Power supply or voltage problems | Seconds | No. Also, `OUT_VOLT` is missing in about 53% of rows |

## Recommended approach

1. **Use each well's own baseline instead of nameplate amps.** Compute something like `CURRENT / rolling-median(CURRENT)` and flag drift as a "pseudo underload/overload".
2. **Normalise for frequency.** Amps change when the drive frequency changes, so separate trend changes from setpoint changes (`FREQUENCY` / `FREC(Hz)`).
3. **Combine current with other variables, as the troubleshooting tables in Takács and API RP 11S do.** For example:
   - Amps down while intake pressure (`PIP`) is low → pump-off or gas
   - Amps up while motor temperature (`TEMP_MOT`) and vibration rise → mechanical problem
   - Amps up while `BSW` rises → heavier fluid or emulsion
4. **Fix the 20 A floor before building features.** Find out whether it stands for "off" or "missing data", and turn it into a separate shutdown flag.
5. **If you want a true amp-chart analysis**, ask for the raw SCADA or VSD current history (1 s to 15 min intervals). That's the data the classification papers below use.

## Literature

**Guidebooks and standards**
- **API RP 11S**, *Operation, Maintenance and Troubleshooting of ESP Installations*. This is the standard reference for interpreting ammeter charts ([listing](https://standards.globalspec.com/std/988038/api-rp-11s)).
- **Takács, G. (2009)**, *Electrical Submersible Pumps Manual*, Gulf Professional Publishing. It has a chapter on amp charts and troubleshooting.
- **Bates et al. (2004)**, "Taking the Pulse of Producing Wells—ESP Surveillance", *Oilfield Review* 16.
- Baker Hughes (Centrilift) *Submersible Pump Handbook*. It's the classic catalogue of amp-chart patterns, but I couldn't find an online copy.

**Papers on current-based diagnosis**
- **Han et al. (2015)**, "Real-Time Monitoring and Diagnosis of ESP", SPE-174873-MS. Diagnosis from current trends.
- **Diagnosis of Operating Conditions of the ESP via Machine Learning (2022)**, *Sensors*, 23(1):279 ([PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC9823322/)). It classifies 24-hour current traces from 20 Brazilian wells into four classes (normal, normal with gas, gas interference, gas lock), reaching about 93% accuracy with SVM. The traces have roughly 344 points per day, which shows the resolution amp-chart classification needs.
- **Han et al. (2025)**, "Fault Diagnosis of ESP Based on Motor Current Signal Analysis and Deep Learning", *SPE Journal* 30(5) ([listing](https://faculty.cup.edu.cn/hanguoqing/lwcg/be18b42756c443b1bc4efe6a53231637.html)).
- **Liu et al. (2024)**, a hybrid LSTM-CNN model for ESP condition prediction, *SPE Journal* 29(5).
- **SPE 229219**, which combines physics-based rules with machine learning for ESP condition monitoring ([JPT summary](https://www.citedrive.com/en/discovery/machine-learning-provides-reliable-esp-condition-monitoring/)).
- **Explainable-AI ESP failure diagnosis (2025)**, *J. Chem. Pet. Eng.* ([link](https://jchpe.ut.ac.ir/article_107377.html)). It uses multivariate field data, which is closer to your dataset.
- **KUNGFU.AI case study** ([link](https://www.kungfu.ai/case-study/advisory-exploration-and-proof-of-concept-with-drilling-technology-leader)). It reports that 24-hour averages still held enough signal to predict failure, and that `PIP` tends to rise before solids-related failures. This is the closest match to your data resolution, though it isn't peer-reviewed.

I only read the full text of the *Sensors* paper. The SPE papers are summaries from listings, so check them on OnePetro before citing.

I can build the per-well baseline amp features and a 20 A shutdown flag in `amperedet.py` if you want.

## You — 2026-11-02T12:34

Is it possible that for the AMPERAGE data can be read as scada? If no the options number 3 seems intereting where I think I'd like to utilize LSTM

## Claude — 2026-11-02T12:35

## Is `AMPERAGE` SCADA data?

**No.** I checked how often each column changes from one day to the next, and `AMPERAGE` follows the well-test columns rather than the sensor columns:

| Column | Changes day to day | Changes on the same day `AMPERAGE` changes |
|---|---|---|
| `AMPERAGE` | ~9% | — |
| `BFPD` (well test) | 9.7% | 58.8% (6× its usual rate) |
| `PIP(PSI)` (well test) | 6.5% | 52.7% (8×) |
| `FREC(Hz)` (well test) | 5.4% | 39.1% (7×) |
| `CURRENT` (sensor) | 54% | — |
| `PRESS_INT` (sensor) | 67% | — |

`AMPERAGE` is also a whole number 86% of the time, and it changes only every 4 days at the median (sometimes 12 days or more). It looks like the amps recorded during a well test and copied forward until the next test.

The columns split into two groups:
- **Sensor (SCADA/gauge), daily values:** `CURRENT`, `FREQUENCY`, `PRESS_INT`, `PRESS_DESC`, `TEMP_INT`, `TEMP_MOT`, `OUT_VOLT`, `VIBRATION`
- **Well test, held constant between tests:** `AMPERAGE`, `BFPD`/`BOPD`/`BWPD`, `BSW`, `PIP(PSI)`, `FREC(Hz)`, `WHP(PSI)`, `GOR`, `GLR`, `API`

So `CURRENT` is your SCADA current. It's still one value per day (probably a daily average or a snapshot), so the earlier point stands: you can analyse trends, but not the patterns inside an amp chart.

## LSTM plan for option 3 (combining current with other variables)

Your existing `lstmanodet.py` trains a separate autoencoder for each variable (Ampere, Volt, TM) on a different CSV, and it uses hard-coded Mac paths. I'd recommend changing it to one **multivariate model**, because the diagnosis comes from how the variables move together.

**1. Inputs (per well, per day)**
- Sensor columns: `CURRENT`, `FREQUENCY`, `PRESS_INT` (pressure at the pump intake), `PRESS_DESC` (discharge pressure), `TEMP_MOT`, `TEMP_INT`, `VIBRATION`
- Amp-chart-style features:
  - `CURRENT` divided by its own 30-day rolling median, as a stand-in for underload/overload
  - `CURRENT` divided by `FREQUENCY`, to separate a real change from a drive setpoint change
  - `PRESS_DESC − PRESS_INT`, the pressure the pump adds
- A shutdown flag where `CURRENT == 20`, with those values masked out rather than fed to the model as real amps
- Optionally the latest well test (`BSW`, `BFPD`), carried forward as context
- I'd leave out `OUT_VOLT`, since it's missing in 53% of rows

**2. Model.** I'd recommend an **LSTM autoencoder trained only on healthy windows**. You have just 105 failure events, which is too few to train a supervised classifier on its own.
- Each window is 30 days of data, and windows that end within 30 days of a failure label are excluded from training.
- Each well is scaled separately, so the model learns relative behaviour rather than absolute amps.
- Training, validation and test sets are split **by well**, so no well appears in two sets and the scores aren't inflated.

**3. Scoring and validation**
- The anomaly score is the reconstruction error, broken down by variable. That breakdown explains each alarm, for example "current falling while motor temperature rises".
- Use the `FAILURE` labels to measure how many failures are caught, how many days before the failure, and how many false alarms each well gets per year.

**4. Optional second stage.** Use the autoencoder's encoding plus the error broken down by variable as features for a small classifier that predicts "failure within the next N days". Class weighting would handle how rare failures are.

One thing to settle first: does `FAILURE` = 1 or 2 stand for a failure type (electrical vs mechanical, for example) or for something else? If it's a type, the second stage could also predict which kind of failure is coming.

I can write this as a new `src/` module with the data preparation, model, training and evaluation, keeping your Streamlit app separate. Should I go ahead?

## You — 2026-11-02T12:38

I would rather using your interpretation on number 3

## Claude — 2026-11-02T12:42

Windows also need to skip days that don't have a baseline yet. I'll add that, plus a threshold-free AUC to the metrics.

## Claude — 2026-11-02T12:43

I built and ran the multivariate LSTM autoencoder. On this daily data it **does not yet beat a simple amp-chart rule**.

## Results (5-fold cross-validation split by well, all 105 failures scored on wells the model never trained on)

| Method | AUC* | Failures caught ≤30 days ahead | Median warning | False alarms per well-year |
|---|---|---|---|---|
| **LSTM autoencoder** | 0.554 | 9.5% | 22 days | 0.38 |
| Rule: current ±20% from its 30-day median | **0.571** | **15.2%** | 20 days | 0.51 |
| Rule: pump off ≥3 of the last 5 days | 0.520 | 30.5% | 30 days | 1.31 |

\*AUC compares the 30 days before a failure with normal days, without needing a threshold. 0.5 means no better than random.

The pump-off rule's 30% catch rate is misleading. It's in alarm on 19% of normal days, so it's mostly noise.

## What I learned along the way

- **The first version was wrong, and I fixed it.** I had scaled each well against its first 180 days, but most wells get a new pump partway through the record, and that version scored no better than random (AUC 0.49). Comparing each day with the same well's previous 60 days instead (a rolling baseline) brought it up to 0.554.
- **The data does hold a weak warning signal, and it fits the literature.** Intake pressure drifting up (AUC 0.58) and current moving away from its normal level (0.57) are the strongest single signs before a failure.
- **The alarm threshold isn't stable.** It ranged from 8 to 39 across the folds, so the model is sensitive to which wells end up in the validation set.
- **My reading of the open questions:**
  - I treated any `FAILURE` > 0 as a failure event.
  - I treated `CURRENT` = 20 A as "pump off" and masked those days out instead of feeding them as real amps.
  - I ignored the zero pressures and the 175° temperature floor as missing readings.

## Bottom line

With one value per day, the LSTM learns to reproduce normal behaviour well, but most pre-failure windows don't look unusual enough to stand out. The precursors are small drifts, which the simple ratio rule already captures. That fits the research from the first question: amp-chart diagnosis needs data recorded every few minutes or seconds.

## Files
- `src/esp_lstm_ae.py`: data preparation, model, cross-validation, the rule baselines and evaluation. Run with `python src/esp_lstm_ae.py`.
- `output/lstm_ae/`: `scores.csv` (daily score, alarms, error per variable), `events.csv` (detected or not, and lead time, per failure), `metrics.json`
- `model/esp_lstm_ae.pt`: final model trained on all wells

The most promising next step is a supervised model that predicts "failure within 30 days". It would take the rolling-deviation features plus the autoencoder's error per variable as inputs, and could combine the weak signals better than any single rule. I can build that, or connect this model to your Streamlit app. Let me know which you'd prefer.

## You — 2026-11-02T12:44

build the supervised 30-day failure model

## Claude — 2026-11-02T12:45

Two spots need cleanup before running: the final-model tree count expression, and how the reference scores are merged in.

## Claude — 2026-11-02T12:47

I built and ran the supervised "failure within 30 days" model. **LightGBM is the best model so far, but the improvement is modest.** One value per day limits how much warning any model can extract from this data.

## Results (5-fold cross-validation split by well, all 105 failures)

Day-level ranking (does the model score the 30 days before a failure higher than normal days?):

| Model | AUC | PR-AUC (base rate 5.5%) |
|---|---|---|
| **LightGBM** | **0.654** | **0.123** (2.2× random) |
| Rule: current ±20% from its 30-day median | 0.571 | 0.078 |
| LSTM autoencoder | 0.554 | 0.077 |
| LSTM classifier | 0.513 | 0.070 |

Failures caught, with every method tuned to raise alarms on the same 5% of normal days:

| Model | Caught ≤30 days ahead | Median warning | False alarms per well-year |
|---|---|---|---|
| **LightGBM** | **24%** | 24 days | **0.80** |
| Amp-ratio rule | 23% | 20 days | 1.27 |
| LSTM autoencoder | 19% | 23 days | 1.26 |
| LSTM classifier | 13% | 15 days | 0.79 |

LightGBM catches about the same number of failures as the simple rule, with about 37% fewer false alarms and a few more days of warning. At tighter alarm budgets (1–2% of normal days) it's roughly tied with the rule.

**The LSTM classifier performed worst.** 105 failure events are too few for it to learn from raw sequences, so the hand-built amp-chart-style features do better.

## What drives the predictions
1. **Run age** (days since the last failure or record start). This is the usual ESP wear-out effect.
2. **Intake pressure drifting up** over 7 days. The same precursor appeared in the autoencoder step.
3. **Intake temperature level and fluctuation**, plus days since the last shutdown.
4. The **autoencoder's anomaly score**, which helps as one input among many even though it was weak alone.
5. Well-test context: `BFPD`, `GLR`, `BSW`, and the 30-day change in test intake pressure.

I also tested the model without run age and the well-test features. AUC barely changed (0.673 vs 0.676), so it isn't just learning well identity or age.

## Caveats
- **I fixed the model size (150 trees) after looking at an out-of-fold diagnostic.** Early stopping on only 11 validation wells stopped at 1–2 trees, so I couldn't use it. Results were flat from 50 to 400 trees, so the optimism is small, but it isn't zero.
- **With 105 events, the catch rates have roughly ±8 percentage points of uncertainty.** Differences of a few points between methods aren't meaningful.
- **I still treated both `FAILURE` = 1 and 2 as failures.** The model catches 35% of type-2 events against 21% of type-1. If 2 means a different failure mode, a separate model per type may be worth it.

## Files
- `src/esp_failure_clf.py`: the features, LightGBM, the LSTM classifier and the evaluation. Run `esp_lstm_ae.py` first, because the autoencoder errors are used as inputs.
- `output/failure_clf/`: `predictions.csv` (daily probability and alarm per well), `events.csv`, `metrics.json`, `feature_importance.csv`
- `model/esp_failure_lgbm.txt`: final LightGBM trained on all wells

The biggest remaining improvement would come from higher-frequency current data, not a better model. With what's here, I can add LightGBM's daily failure risk per well to your Streamlit app if you want.

## You — 2026-11-02T12:48

add it to the Streamlit app

## Claude — 2026-11-02T12:48

`app.py` is actually a FastAPI service. There are four Streamlit apps, so I'll check which one fits.

## You — 2026-11-02T12:48

Base directory for this skill: /tmp/claude-1000/bundled-skills/2.1.316/3c4ad4dc3d0af58d77db885803aca8e5/dataviz

# Data Visualization

A chart is **read by people and executed by you**. This skill turns "make it look
good" into a procedure with checks, so the result is right by construction rather
than by taste.

**The method here is design-system-agnostic.** Nothing in the procedure, the form
heuristic, the six checks, or the mark specs is specific to one product. A design
system supplies a small set of *parameters* (its ramps, a categorical order, a
diverging pair, a status palette, a texture, its surfaces, its filter components);
the method consumes them unchanged. A **validated default palette** is the
reference instance, fully specified in `references/palette.md`. To target your
brand, read that file's structure and substitute its values - touch nothing else.

> The single most important habit: **the color part is computable, so compute it.**
> Never eyeball whether a palette is colorblind-safe - run `scripts/validate_palette.js`.

## The procedure - do these in order

Color comes LAST. Most bad charts pick colors first.

1. **Pick the form.** What is the data's job - magnitude, identity, polarity, a
   single headline, change-over-time? The job picks the chart type, and sometimes
   the answer is *not a chart* (a stat tile or hero number). -> `references/choosing-a-form.md`
2. **Assign color by the job it does.** Categorical (identity), sequential
   (magnitude), diverging (polarity), or status (state) - each has one rule.
   Assign categorical hues in fixed order, never cycled. -> `references/color-formula.md`
3. **VALIDATE the palette - run the script, don't reason about Delta E.**
   `node scripts/validate_palette.js "<hex,hex,...>" --mode light` (relative to
   this skill's base directory - or load it as `<script type="module">` in the
   chart's own page, where it reads
   `data-palette` off `<body>` and logs a `console.table` report). It returns
   pass/fail on the lightness band, chroma floor, adjacent-pair CVD separation,
   the normal-vision floor, and contrast. Fix anything that FAILs before continuing. Re-run for
   `--mode dark` with that mode's surface.
4. **Apply mark specs & spacers.** Thin marks, 4px rounded data-ends anchored to
   the baseline, 2px lines, >=8px markers, a 2px surface gap between fills (stacked
   segments and adjacent bars alike) and a 2px surface ring on overlapping marks,
   selective direct labels. -> `references/marks-and-anatomy.md`
5. **Add the hover layer - by default.** An HTML/SVG chart *is* interactive; ship
   a crosshair+tooltip on line/area and a per-mark hover tooltip on bar/dot/cell.
   The only form that skips it is a bare stat tile with no plot. Hit targets bigger
   than the mark; filters in one row above the charts. -> `references/interaction.md`
6. **Final accessibility pass.** For >= 2 series a legend is always present and <= 4
   are also direct-labeled (a single series needs no legend box - the title names
   it), so identity is never color-alone; a table view exists; dark mode is **selected** - its own
   steps from the same ramps, validated against the dark surface, not an automatic
   flip; texture is available for the CVD/print/forced-colors case.
7. **Render it and look at it.** The validator checks color, not layout - open or
   screenshot the output and eyeball it for label collisions, geometry, and overflow
   before calling it done.

Then check the result against **`references/anti-patterns.md`** - it is the catalog
of what goes wrong. If your chart matches an entry, it's wrong.

## Non-negotiables (true in every design system)

- **Assign categorical hues in fixed order, never cycled.** A 9th series is never a
  generated hue - it folds into "Other," small multiples, or composite encoding.
- **One axis.** Never a dual-axis chart (two y-scales). Two measures of different
  scale -> two charts, small multiples, or indexed to a common base. *(This is the
  #1 chart mistake - see anti-patterns.)*
- **Color follows the entity, never its rank.** A filter that changes the series
  count must not repaint the survivors.
- **Sequential = one hue, light->dark. Diverging = two hues + a neutral gray
  midpoint.** Never a rainbow; never a hue at the diverging midpoint.
- **Run the validator before shipping any categorical palette.** CVD Delta E >= 8 is the
  target (OKLab ×100); 6-8 is a floor that is legal ONLY with secondary encoding. A
  normal-vision floor below 15 is a hard FAIL - full-color readers can't tell the
  pair apart; re-step it on the adjacent pairlist (secondary encoding does not excuse
  this one); under `--pairs all` cut series or facet instead - see check 4. A contrast WARN
  obligates visible labels or a table view - it is not dismissable.
- **Thin marks; a legend always present for >= 2 series (none for one), with
  selective direct labels (never a number on every point); recessive grid/axes.**
  One exception, legend kept: the tall segmented bar (*Figures* in `references/marks-and-anatomy.md`).
- **Text wears text tokens, never the series color** - values, labels, and legends
  stay in primary/secondary/muted ink; a colored mark beside them carries identity.
- **Status colors are reserved** (good/warning/serious/critical) and never reused
  for "series 4"; they ship with an icon + label, never color alone.

## Plugging in a design system

The method is invariant; only these parameters change per system. The reference
instance - every value filled in - is `references/palette.md`.

| Parameter | What the system provides |
|---|---|
| **Ramps** | the hue scales (named steps) the palette draws from |
| **Categorical theme** | the fixed hue order (a named theme); default + alternates |
| **Sequential hue** | the default single hue for magnitude |
| **Diverging pair** | two warm/cool poles + a neutral midpoint |
| **Status palette** | good / warning / serious / critical - steps distinct from categorical |
| **Texture fill** | one directional hand-drawn fill, used at 45° / 135° |
| **Surfaces** | light & dark chart-surface colors (the validator needs these) |
| **Filter controls** | date-range & dimension controls (behavioral spec in `interaction.md`) |

To onboard a new system: fill those rows, feed its ramps to the validator, and let
it snap each slot to the nearest passing step. Structure and rules stay as written.

## Reference files

| File | What it answers |
|------|-----------------|
| `references/choosing-a-form.md` | Which chart type / is it even a chart? |
| `references/color-formula.md` | The four jobs, the six checks, snap-to-passing |
| `references/marks-and-anatomy.md` | Mark specs, spacers, labels, figures, hero number |
| `references/interaction.md` | Tooltips & hover, filters & time ranges |
| `references/components.md` | The pieces a chart is made of - build each in plain HTML |
| `references/anti-patterns.md` | **What goes wrong - check every chart against this** |
| `references/palette.md` | **The reference palette instance** - every parameter, filled in; swap for your brand's |
| `scripts/validate_palette.js` | Runnable six-checks validator (run it; don't eyeball) |


## Claude — 2026-11-02T12:49

I'll add an inference path to both modules: AE errors from the saved autoencoder, and a `score_wells()` function that the app can call.

## Claude — 2026-11-02T12:50

The results reproduce exactly and the metadata is saved. Now I'll time the inference path on the full dataset.

## Claude — 2026-11-02T12:50

Inference takes about 2 seconds for all 95 wells and also works without a `FAILURE` column. I'll add per-day risk drivers through LightGBM's built-in feature contributions, then write the app.

## Claude — 2026-11-02T12:51

Now I'll test the app headlessly with Streamlit's `AppTest`, checking the fleet view, well selection, the threshold slider and the drivers chart.

## Claude — 2026-11-02T12:51

Everything works with no exceptions. I'll swap out the deprecated `use_container_width` argument, then start the app and screenshot it to check the layout.

## You — 2026-11-02T12:51

[Image: original 1400x2600, displayed at 1077x2000. Multiply coordinates by 1.30 to map to original image.]

## Claude — 2026-11-02T12:51

That was the loading skeleton, so the screenshot fired before the websocket delivered content. I'll retry with a real-time delay.

## Claude — 2026-11-02T12:53

The chart works. One fix: the "Failure" label overlaps the threshold label at the top left. I'll add headroom above the threshold line and move its label to the bottom right.

## Claude — 2026-11-02T12:54

The threshold label now sits on the risk line at the right edge. I'll drop the annotation and name the dashed line in the panel title instead.

## Claude — 2026-11-02T12:54

I added the failure-risk model as a new Streamlit app, `risk_app.py`, instead of changing an existing one. Your current apps (`isoapp.py`, `amperedet.py`, `lstmanodet.py`) all expect the older CSV layout (`Well`, `Date`, `Ampere`), so none of them can read `wells_data_final.csv`. `app.py` turned out to be a FastAPI service, not Streamlit. I left all of those untouched.

Run it from the project folder:

```bash
streamlit run risk_app.py
```

I tested it with Streamlit's headless test runner (fleet view, well switching, threshold slider, date picker, the toggle to show all wells) with no errors. I also ran it in a browser and checked screenshots, which led to three fixes. One mattered: wells whose record ends on their failure day were showing at the top of the list as "Alarm", so they now show as "⚫ Pump off". The other two were overlapping chart labels.

**What's in the app**
- **Sidebar:** uses `data/wells_data_final.csv` by default, or a CSV you upload in the same layout (the `FAILURE` column is optional). It also has the risk threshold slider, set by default to the cross-validated value of 0.63, and a toggle for showing recorded failures on the charts.
- **Fleet:** counts of wells in alarm or elevated, the cross-validation results (24% of failures caught, 0.84 false alarms per well-year), and a table of every well ranked by its latest risk with a status column (🔴 Alarm / 🟡 Elevated / 🟢 Normal / ⚫ Pump off).
  - By default only the 67 wells still reporting on the latest date (2021-05-25) are shown; the toggle brings back all 95.
  - With the defaults, 2 of the 67 are in alarm: MECD2B and UVG14O, the latter in alarm on 17 of the last 30 days.
- **Well detail:** a risk chart with the threshold line, shaded alarm days and failure markers. Below it are separate panels for motor current, intake pressure and intake temperature, sharing the same time axis.
- **What drives the risk:** pick any date to see which features pushed that day's score up or down, with plain-language names such as "Intake pressure vs baseline, 7-day mean".

Scoring all 95 wells takes about 2 seconds and is cached, and moving the threshold slider doesn't trigger a re-score.

**Caveats**
- The risk number is a ranking score, not a probability that the pump fails, and the app says so. The training weighted failures heavily, so the scores run higher than real odds.
- On wells the model was trained on, the app's scores track the fair cross-validation scores fairly closely but not exactly (correlation 0.78). Use the cross-validation numbers to judge how well it works.

**Files**
- New: `risk_app.py` and `model/esp_failure_lgbm.json` (feature list, threshold, test results)
- Changed: `src/esp_failure_clf.py` gained `score_wells()` for use outside the app, and `src/esp_lstm_ae.py` gained `ae_errors()`. I re-ran training to confirm the results came out the same.
- The 30-day horizon and the ≥3-of-5-days alarm rule are fixed in the saved model files.

`git status` shows changes to `src/datascratch.py` that I didn't make. Did you edit it during the session? I've left it alone. Nothing is committed yet, so tell me if you want a commit.

## You — 2026-11-02T12:56

what to run?

## Claude — 2026-11-02T12:56

The models are already trained, so you only need to start the app. Run this from the project folder with your venv active:

```bash
cd /home/rianr/pypro/myvenv/esp_forecasting
source ../bin/activate.fish      # your login shell is fish; in bash/zsh use: source ../bin/activate
streamlit run risk_app.py
```

Then open the link it prints, usually http://localhost:8501.

**If the data changes, retrain first.** The order matters, because the second script uses the autoencoder's output:

```bash
python src/esp_lstm_ae.py        # step 1: LSTM autoencoder (~1–2 min on your GPU)
python src/esp_failure_clf.py    # step 2: 30-day failure model, saves what the app uses
streamlit run risk_app.py        # step 3: the app
```

To score new data without retraining, upload a CSV in the same layout as `wells_data_final.csv` from the app's sidebar.

## You — 2026-11-02T12:59

ok please explain in details how everything works

## Claude — 2026-11-02T13:00

# How the ESP failure-risk system works

## 1. The big picture

```
wells_data_final.csv (one row per well per day)
        │
        ▼
┌───────────────────────── src/esp_lstm_ae.py ─────────────────────────┐
│ engineer()        clean the data, flag pump-off days, add 3 features  │
│ mark_exclusion()  mark the days around each failure                   │
│ scale_per_well()  compare each day with the well's own previous 60 d  │
│ LSTM autoencoder  learns "normal", scores how abnormal each day is    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │ anomaly score + error per variable
                               ▼
┌──────────────────────── src/esp_failure_clf.py ──────────────────────┐
│ build_features()  64 rolling features per day                         │
│ LightGBM          predicts "will this well fail in the next 30 days?" │
│ threshold + 3-of-5-days rule → alarm                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               ▼
                    risk_app.py (Streamlit dashboard)
```

There are two models. The **autoencoder** learns what normal looks like without using failure labels. The **LightGBM classifier** learns from the 105 recorded failures what the weeks before a failure look like, and it uses the autoencoder's output as one of its inputs.

---

## 2. What the data looks like (and why it matters)

Profiling the CSV shaped every design decision:

| Finding | Consequence |
|---|---|
| One row per well per day (95 wells, 821 days) | Classic amp-chart patterns that happen within a day (gas spikes, cycling) can't be seen, so the model works with **trends over days** |
| Columns fall into two groups: **sensor data** (`CURRENT`, `PRESS_INT`, `TEMP_MOT`…, changes daily) and **well-test data** (`AMPERAGE`, `BFPD`, `BSW`, `PIP(PSI)`…, held constant between tests) | `CURRENT` is the real SCADA amps. `AMPERAGE` is a well-test reading carried forward, so it isn't used as a sensor |
| `CURRENT` is exactly 20 A in 22% of rows | Treated as **"pump off"**, not as a real value of 20 A |
| Other floor values: `PRESS_INT` = 0, temperatures ≤ 175, frequency ≤ 25 | Treated as missing readings |
| 105 failure events (`FAILURE` = 1 or 2), at least one in every well | Any value > 0 counts as a failure, so the two types are combined |

---

## 3. Step 1: preparing the data (`esp_lstm_ae.py`)

### `engineer()`: cleaning and amp-chart features
1. Adds an `OFF` flag (1 when `CURRENT` ≤ 20 A) and blanks out the sensor values on those days.
2. Turns the floor and zero values listed above into missing values.
3. Adds three features based on amp-chart reasoning:
   - **`AMP_RATIO`** = current ÷ the well's own 30-day median current. Amp charts judge underload and overload against the motor's nameplate amps, which this data doesn't have, so the well's recent normal current stands in for it. A value of 0.8 means 20% underload.
   - **`AMP_PER_HZ`** = current ÷ frequency. It separates a real change in load from an operator changing the drive speed.
   - **`PUMP_DP`** = discharge pressure − intake pressure, the pressure the pump is adding.
4. Records which values were real measurements (columns `M_*`), then fills gaps by carrying the last value forward. The model is never penalised for failing to reproduce a filled-in value.

### `mark_exclusion()`: separating normal days from failure periods
- Marks **30 days before to 60 days after** each failure as `EXCLUDED`. The days before are the warning period being predicted, and the days after cover the workover and restart, which aren't normal operation.
- Computes `T_TO_FAIL`, the number of days until the next failure, which is used for labels and evaluation.

### `scale_per_well()`: the rolling baseline (the most important design choice)
Each value is converted into "how far it is from this well's normal level":

```
z(t) = (x(t) − median of days [t−90 … t−31]) / IQR of the same days
```

The comparison period is the 60 days that ended 30 days ago. Two reasons:
- Wells differ a lot (one runs at 70 A, another at 25 A), so absolute values aren't comparable between wells.
- **The first version compared each well against its first 180 days, and it failed (AUC 0.49, i.e. random).** Once a pump fails and is replaced, the new pump behaves differently from the old baseline, so everything afterwards looked abnormal. A rolling baseline resets on its own after a pump change.

Two safeguards: the IQR has a floor (5% of the median) so that flat, carried-forward series don't divide by almost zero, and values are clipped to ±10. The first ~90 days of each well have no baseline yet (`HAS_BASE` = False), so they're skipped.

---

## 4. Step 2: the LSTM autoencoder (`esp_lstm_ae.py`)

### Idea
Train a network to **reproduce normal 30-day windows**. Afterwards, a window it reproduces badly is one that doesn't look like normal operation.

### Input
For each day, a 30-day window of 11 variables: 7 sensors (current, frequency, intake and discharge pressure, intake and motor temperature, vibration), the 3 engineered features, and the `OFF` flag. The array shape is `(windows, 30, 11)`.

### Architecture (`LSTMAE`)
```
input (30 days × 11) → 2-layer LSTM (64 units) → last hidden state → Linear → 16-number summary
16-number summary → repeated 30 times → 2-layer LSTM (64 units) → Linear → reconstruction (30 × 11)
```
The 16-number bottleneck forces the network to learn the typical patterns instead of copying its input.

### Training (`train()`)
- Only on **normal windows**, meaning no day in the window falls in an excluded period.
- The loss is a **masked mean squared error**: only real measurements count, not filled-in values.
- Adam optimiser (learning rate 1e-3), gradient clipping, and early stopping after 5 epochs without improvement on the validation wells.

### Scoring (`score()`)
- **Anomaly score** = reconstruction error over the **last 7 days** of the window. Using the whole 30 days would dilute recent changes.
- It also records the **error for each variable** (`ERR_CURRENT`, `ERR_PRESS_INT`…), which shows *what* looks abnormal.
- Each day is scored using the window that ends on that day, so a score never uses future data.

### Alarm
- Threshold = 99th percentile of anomaly scores on normal days in the validation wells.
- `persist()` raises an alarm only when the threshold is exceeded on **≥3 of the last 5 days**, so one noisy day doesn't trigger it.

### Result
AUC 0.554, catching 9.5% of failures. On its own it's weak (the simple ±20% amp rule did better, at 0.571), but its anomaly score turned out to be a useful input to the next model.

---

## 5. Step 3: the supervised 30-day model (`esp_failure_clf.py`)

### Labels
| Day type | Label |
|---|---|
| 1–30 days before a failure | **1** (about 5.5% of days) |
| Normal (not excluded) | **0** |
| The failure day and the 60 days after it | not used |

### Features (`build_features()`, 64 in total, all computed from past data only)

| Group | Features | Amp-chart reasoning |
|---|---|---|
| Deviation from the well's normal level, for each of the 10 continuous variables | today's value (`_z`), 7-day mean, 7-day standard deviation, 7-day trend (this week's mean minus last week's) | Drift, step changes, erratic behaviour |
| Shutdowns | share of days off in the last 7 and 30 days, number of shutdowns in 30 days, days since the last shutdown | The daily version of "cycling" and "pump-off" on an amp chart |
| Run age | days since the last failure | ESP wear-out |
| Well test | `BSW`, `GLR`, `BFPD`, their 30-day changes, 30-day change in test `PIP` | Changes in the fluid and inflow |
| Autoencoder | anomaly score plus the 11 per-variable errors | Everything that "looks abnormal" |

`_days_since()` counts days since the last event within each well, and is used for run age and days since the last shutdown.

### LightGBM settings and why
```python
learning_rate=0.02, num_leaves=7, min_child_samples=500,
subsample=0.7, colsample_bytree=0.5, reg_lambda=10, n_estimators=150
```
- **Heavy regularisation**, because there are only 105 failures. Small trees with at least 500 days per leaf can't memorise individual wells.
- **`scale_pos_weight`** = number of normal days ÷ number of warning days, which balances the rare positive class. It's also why the risk scores run high: they rank days but aren't probabilities.
- **The tree count is fixed at 150** because early stopping on only ~11 validation wells was too noisy and stopped after 1–2 trees. Accuracy is flat between 50 and 400 trees.

### The LSTM classifier (comparison only)
The same 30-day windows go into an LSTM followed by a small classification layer, with class weighting. It scored worst (AUC 0.513) because 105 events are too few to learn from raw sequences. The app doesn't use it.

---

## 6. How the evaluation works (why the numbers are trustworthy)

### Cross-validation split by well
```
95 wells → 5 folds of ~19 wells
for each fold:
    test wells: never seen in training
    remaining wells → 85% train, 15% validation (to set the threshold)
```
Every well is scored by a model that never saw it, so all 105 failures are evaluated out of sample. Splitting by date instead would let the model learn each well's characteristics and inflate the scores.

### Metrics (`evaluate()`)
| Metric | Meaning |
|---|---|
| **AUC** | How well the score ranks warning days above normal days (0.5 = random) |
| **PR-AUC** | Precision on the rare class; compare it with the 5.5% base rate |
| **Failures caught ≤30 d** | Share of failures with at least one alarm in the 30 days before |
| **Median warning** | How many days before the failure the first alarm came |
| **False alarms per well-year** | Number of separate alarm episodes on normal days, scaled to a year |

### Fair comparison
Every method was tuned to raise alarms on the same 5% of normal days:

| | Caught | False alarms / well-year |
|---|---|---|
| LightGBM | 24% | 0.80 |
| Amp-ratio rule | 23% | 1.27 |
| Autoencoder | 19% | 1.26 |

### Final models
After cross-validation, both models are retrained on all wells and saved:
- `model/esp_lstm_ae.pt`
- `model/esp_failure_lgbm.txt`
- `model/esp_failure_lgbm.json`: feature list, threshold (the mean of the per-fold thresholds = 0.63), the 3-of-5 rule and the cross-validation results

---

## 7. How the app works (`risk_app.py`)

```
CSV (default file or upload)
  → score()  [cached]  → fc.score_wells(raw, explain=True)
                           ├ prepare(): engineer → mark_exclusion → scale_per_well
                           │            → build_features + ae_errors (saved autoencoder)
                           ├ booster.predict()                    → RISK per day
                           └ booster.predict(pred_contrib=True)   → drivers per day
  → threshold slider → fc.alarm() → ALARM (instant, no re-scoring)
  → fleet_table() → well_figure() → drivers_figure()
```

- **Caching:** the slow part (scoring, about 2 s) runs once per file. Moving the threshold slider only recomputes the alarm rule.
- **Status logic** (based on each well's last day): ⚫ pump off on that day → 🔴 alarm rule triggered → 🟡 risk above 85% of the threshold → 🟢 normal.
- **"Only wells reporting on the latest date"** hides wells whose record ended earlier, so wells that have already stopped don't rank as "current" risks.
- **Drivers chart:** LightGBM's built-in per-day contributions in log-odds (the same idea as SHAP values). Orange bars raise that day's risk and blue bars lower it.
- **Sensor panels** show the raw, unscaled values, with pump-off days left blank.

---

## 8. Limitations
1. **Daily resolution is the main ceiling.** It catches about a quarter of failures at under 1 false alarm per well-year. Real amp-chart diagnosis needs data every few minutes or seconds.
2. **The risk score isn't a probability.** Read it relative to the threshold.
3. **Failure types are combined.** If `FAILURE` = 2 means a different failure mode, a separate model per type could help.
4. **About ±8 percentage points of uncertainty** on the catch rate with 105 events.
5. **On wells used in training**, the app's scores are close to but not identical to the out-of-sample ones (correlation 0.78).

## 9. Where to tune
| What you want | Where to change it |
|---|---|
| Fewer or more alarms | App slider; `--alarm_quantile` in `esp_failure_clf.py` |
| A different prediction horizon (e.g. 14 days) | `PRE_DAYS` in `esp_lstm_ae.py` (also sets `HORIZON`) |
| A different pump-off definition | `CURRENT_FLOOR` |
| A different baseline window | `BASE_LAG`, `BASE_REF` |
| A different alarm rule | `persist_k` / `persist_n` in the saved JSON, or `persist()` |

After changing any of these, retrain in order: `esp_lstm_ae.py`, then `esp_failure_clf.py`.

I can also publish this explanation as a formatted page you can share with your team.

## You — 2026-11-02T13:00

what is esp_failure_clf?

## Claude — 2026-11-02T13:00

`esp_failure_clf.py` (in `src/`) is the script that builds and runs the **30-day failure prediction model**. The name stands for **ESP failure classifier**. For each well and each day it answers a yes/no question: will this pump fail in the next 30 days?

It does three jobs:

**1. Training and testing** (when you run `python src/esp_failure_clf.py`)
- Labels each day: **1** if a failure follows within 30 days, **0** if it's a normal day.
- Builds 64 features per day. These include how far current, intake pressure and temperatures have drifted from the well's normal level, how often the pump shut down recently, run age, well-test changes, and the autoencoder's anomaly score.
- Trains a **LightGBM** model and checks it with 5-fold cross-validation, where each fold tests on wells the model never saw. That's where the 24% caught / 0.84 false alarms per well-year figures come from.
- It also trains an LSTM classifier for comparison. That one did worse, so the app doesn't use it.

**2. Saving the final model**
- `model/esp_failure_lgbm.txt`: the trained LightGBM model
- `model/esp_failure_lgbm.json`: the feature list, the alarm threshold (0.63) and the test results
- `output/failure_clf/`: daily predictions, per-failure results and feature importance

**3. Scoring new data** (used by the app)
- `score_wells()` takes a CSV, prepares it the same way as in training, and returns a daily **risk score** and **alarm** for every well. `risk_app.py` calls this function.

It depends on `esp_lstm_ae.py`, which provides the data cleaning, the rolling baseline and the autoencoder. That's why `esp_lstm_ae.py` has to run first when you retrain.

## You — 2026-11-02T13:01

can you save all this conversation?

## Claude — 2026-11-02T13:02

I saved a written summary of the session to `docs/esp_failure_risk_notes.md` in your project. It covers:
- The amp-chart research verdict and the literature list with links
- What I found in the data (why `AMPERAGE` isn't SCADA, the 20 A pump-off floor, the failure labels)
- How the pipeline and both models work
- All the results tables and caveats
- The files, how to run and retrain, and where to tune
- Open questions and next steps

**For the full conversation word for word**, Claude Code already stores it here:
```
~/.claude/projects/-home-rianr-pypro-myvenv-esp-forecasting-src/0838d459-6096-43d5-a9a2-f86057d95fdf.jsonl
```
That file is raw JSON and not easy to read. To get a readable copy, type `/export` in this session; it saves the conversation as a text file or copies it to the clipboard. Later, `claude --resume` reopens this session.

Nothing is committed to git yet, including the new code, models and `docs/`. I can commit them if you want; I'd leave out `src/datascratch.py`, which has changes I didn't make.

## You — 2026-11-02T13:04

can current be read as scada and then the pattern classified using amp card like

## Claude — 2026-11-02T13:05

Short answer: **yes, `CURRENT` is SCADA**, but only at one value per day. So amp-card-style classification works if each "card" is a **30-day window of daily values** instead of a 24-hour trace. Before recommending it, I'll prototype that classification and check whether the patterns actually relate to failures.

## Claude — 2026-11-02T13:05

Yes. `CURRENT` behaves like SCADA data: it changes on 54% of days, and 32% of its values have more than 2 decimals, which suggests daily averages computed from higher-frequency readings. But you only get **one point per day**. So amp-card classification works if each "card" is a **30-day window of daily values** rather than a 24-hour trace, and only for patterns that build up over days.

I prototyped this on all wells with rules adapted from the classic amp-card patterns, then checked how often a failure followed within 30 days for each pattern (normal rate: 5.4%):

| Daily amp-card pattern | Closest classic pattern | Share of days | Failure within 30 d | Times the normal rate |
|---|---|---|---|---|
| **Cycling** (≥3 shutdowns in 30 d) | Repeated restarts / underload trips | 1.6% | 9.2% | **1.7×** |
| **Erratic / fluctuating** | Gas interference, unstable inflow | 4.6% | 9.2% | **1.7×** |
| **Step change** (sudden level shift ≥15%) | Sudden load change, mechanical event | 0.6% | 9.0% | **1.7×** |
| Gradual rise (>+10% over 30 d) | Overload trend: scale, sand, emulsion | 4.1% | 5.6% | 1.0× |
| Pump down (>50% of days off) | Shutdown | 19.1% | 5.5% | 1.0× |
| Normal (stable) | Normal | 67.3% | 5.0% | 0.9× |
| Gradual decline (<−10% over 30 d) | Underload trend: wear, falling inflow | 2.7% | 3.7% | 0.7× |

**What this shows:**
- **It works as a way to describe a well.** Cycling, erratic and step-change patterns come before failures about 1.7× as often as normal, which fits amp-card reasoning.
- **It's weak as a warning on its own.** Those three patterns cover only 7% of days. The day before a failure, 49 of the 105 failed wells still looked "Normal (stable)", because the warning signs show up in intake pressure and temperature, not only in current. That's why the LightGBM model, which reads current together with the other variables, ranks better.
- **The match to the classic patterns is approximate.** "Erratic" on daily values might be gas interference, but it could also be unstable inflow or data noise. You can't confirm it without the high-frequency trace.
- **The rule thresholds are my first guesses** (5% day-to-day jitter, ±10% trend, ≥3 shutdowns) and haven't been tuned or checked against an engineer's labels.

The best way to use it would be:
1. Show the **amp-card pattern as a status label** in the app, next to the risk score. That gives engineers a familiar reading such as "erratic" or "cycling".
2. **Feed the pattern into LightGBM as an extra input** and check whether it improves the model.

The prototype is a standalone script at `/tmp/claude-1000/-home-rianr-pypro-myvenv-esp-forecasting-src/0838d459-6096-43d5-a9a2-f86057d95fdf/scratchpad/ampcard_proto.py` and isn't part of the project yet. Should I add it to the project and the app?

## You — 2026-11-02T13:06

can you use autoencoder to learn the current pattern and use the classification based on paper/books/literature for that

## Claude — 2026-11-02T13:06

I'll first pin down the actual amp-chart pattern classes in the literature, so the classification is grounded in the sources and not in my own rules.

## Claude — 2026-11-02T13:08

Some cards had no baseline partway through a long shutdown. I'll carry the baseline forward within each well.

## Claude — 2026-11-02T13:09

The autoencoder trained in about 20 seconds. Before interpreting the clusters, I'll look at the median card shape for each one.

## Claude — 2026-11-02T13:15

I built it as `src/esp_ampcard.py`. It works, and the patterns are recognisable amp-card classes, but the result is mixed. The autoencoder is genuinely useful for the running-current shapes. The strongest failure signs are the shutdown sequences, and those come from rules on pump-off timing, not from the autoencoder.

## How it works
1. **The card:** each well-day gets a 30-day window of current divided by the well's normal level (its rolling baseline), plus a pump-off channel. That gives 47,291 cards.
2. **Shutdown sequences** are classified by rules on pump-off timing. These follow Cunningham & Riling's Figs. 5–6 (pumped off and tripped on undercurrent; tripped on underload, then a restart attempt).
3. **Running shape** comes from a 1-D convolutional autoencoder trained only on running days (pump-off days are hidden from it). It compresses each card to 8 numbers, and clustering those gives 10 shape families.
4. **Naming:** each family is named from the shape of its median card (steady, gradual rise, sudden rise, recovering…) and mapped to the literature class (normal, high or low amperage).
   - Two exceptions are handled by rules. Erratic loading is defined by day-to-day jitter, which the autoencoder smooths away. And when the frequency set-point moved but current per Hz stayed normal, the card is labelled a speed change, not overload.
5. **Likely cause** follows the literature pairing. For example, low amps with falling intake pressure points to the well being pumped off, and low amps with stable intake pressure points to a problem inside the pump.

## Results (normal rate of a failure within 30 days: 5.6%)

| Amp-card class | Literature basis | Share of cards | Times the normal failure rate |
|---|---|---|---|
| **Cycling** (≥3 trips in 30 d) | Pump-off / gas-lock cycling | 2.3% | **2.5×** |
| **Gassy / erratic loading** | Gas interference | 6.0% | **1.5×** |
| **Recent trip** (ran, now off) | Undercurrent/underload shutdown | 4.7% | **1.4×** |
| **Overload – gradual rise** (found by the autoencoder) | High amperage trend: scale, emulsion, drag | 2.8% | **1.3×** |
| Normal – steady at baseline | Normal operation | 50% | 1.0× |
| Overload – steady above baseline | High amperage | 7.1% | 0.9× |
| Extended shutdown / restart after shutdown | — | 19% | 0.6–0.8× |
| Speed change (VSD) | No fixed-speed equivalent | 6.5% | 0.5× |

Cards the autoencoder reconstructs badly (the top 1% of reconstruction error) are followed by a failure 1.85× as often as typical cards.

## What I learned building it
- **The first version learned the wrong thing.** Its clusters were organised by when the pump switched off, not by current shape. Training only on running days fixed that.
- **Without frequency normalisation, "overload" was really speed-ups.** One cluster ran a steady 1.4× above baseline and almost never failed. That's an operator raising the drive speed, so the level classes now use current per Hz, plus a separate speed-change class.
- **The autoencoder found something the rules got wrong.** The rules called a sudden step in current "erratic". Shape prototypes separate steps, ramps and recoveries properly.

## Limits
- **Power fluctuations and true gas locking can't be classified.** Both need data recorded every few seconds or minutes.
- **The small clusters (sudden rise, sudden drop) show 0 failures.** They're probably baseline artefacts after pump changes, and they're too small to interpret.
- **The rule thresholds are my reading of the literature**, not values from API RP 11S, which I couldn't access in full.

## Outputs
- `output/ampcard/prototypes.png`: the median card for each shape family. Open this first.
- `output/ampcard/cards.csv`: class, shape, likely cause and unusual flag for every card
- `output/ampcard/class_vs_failure.csv`, `clusters.csv`
- `model/esp_ampcard_ae.pt`

The next step would be adding the amp-card class and likely cause to `risk_app.py` next to the risk score, and testing it as an extra input to LightGBM. Should I do that?

**Sources**
- [Cunningham & Riling, Submersible Electrical Pump Ammeter Chart Interpretation (SWPSC)](https://www.swpshortcourse.org/paper/submersible-electrical-pump-ammeter-chart-interpretation)
- [SWPSC paper page](https://swpshortcourse.org/node/15701)
- [Takács, Electrical Submersible Pumps Manual (Elsevier)](https://shop.elsevier.com/books/electrical-submersible-pumps-manual/takacs/9781856175579)
- [ESP Operation and Troubleshooting Guide (Scribd)](https://www.scribd.com/document/588332251/PTP-06-2-Handout)
- [Nodal-analysis ESP monitoring in multiphase flow (MDPI)](https://www.mdpi.com/2076-3417/12/6/2825)
- [Journal of Energy and Power Technology, ESP reliability](https://www.lidsen.com/journals/jept/jept-04-04-036)
- [ESP Failure Prevention Guide (iFactory, vendor)](https://ifactoryapp.com/industries/oil-and-gas/esp-electrical-submersible-pump-failure-prevention)

## You — 2026-11-02T13:16

save all the conversation

## Claude — 2026-11-02T13:16

I've updated `docs/esp_failure_risk_notes.md` with the amp-card work from this last part of the session. The new **section 7** covers:
- Whether `CURRENT` counts as SCADA data, and the first rule-only prototype
- The literature classes, mainly Cunningham & Riling's figures, plus API RP 11S and Takács
- How the two-tier classifier works (shutdown rules plus the autoencoder for running shape)
- The four iterations and what each one fixed
- The results table, limits and output files

The next steps (section 8) now include adding the amp-card classes to the app and checking the thresholds against API RP 11S.

For a word-for-word copy of the whole conversation, type `/export` in this session. Claude Code also keeps the raw transcript at:
```
~/.claude/projects/-home-rianr-pypro-myvenv-esp-forecasting-src/0838d459-6096-43d5-a9a2-f86057d95fdf.jsonl
```

Nothing is committed to git yet. The new files are `docs/`, `risk_app.py`, the three `src/esp_*.py` modules, `model/` and `output/`. I can commit them, leaving out `src/datascratch.py`, which has changes I didn't make.

## You — 2026-11-02T13:17

add the amp card class to the streamlit app

## Claude — 2026-11-02T13:18

The training results reproduce, with tiny share differences from GPU nondeterminism. Now I'll check that the inference path gives the same classes as the training run.

## Claude — 2026-11-02T13:18

Inference matches the training run 100%. It takes about 12 seconds for all wells, and the app caches the result. Now the app itself:

## Claude — 2026-11-02T13:19

No errors across wells and dates, including the first dates where no card exists yet. Now a visual check in the browser:

## Claude — 2026-11-02T13:21

Both fixes worked. I'll stop the test server, remove the temporary screenshots, and add the new app features to the notes file.

## Claude — 2026-11-02T13:21

The amp-card classes are now in the Streamlit app. Run it the same way as before:

```bash
streamlit run risk_app.py
```

I tested it with Streamlit's headless test runner, switching between wells and picking dates at the start, middle and end of each record, with no errors. I also checked it in a browser and fixed two display issues: a misleading zoom on the card chart and labels that squeezed the panels.

**What's new**
1. **Fleet table:** a new **"Amp card"** column showing each well's current class.
   - With the defaults, most of the 67 active wells read "Normal – steady at baseline" (38).
   - The rest are extended shutdowns (7), speed changes (5), steady above baseline (5), gassy/erratic (3) and gradual rise (3), plus a handful of others.
2. **Well chart:** a new **amp-card class panel** under the risk curve.
   - Each class has its own lane, labelled with how often a failure follows it within 30 days compared with normal, e.g. "Recent trip (1.4×)", "Overload – gradual rise (1.3×)".
   - Hovering over a day shows its class and likely cause.
3. **"Selected day: amp card and risk drivers"**, side by side:
   - Left: that day's 30-day card (current ÷ the well's normal level, pump-off days shaded grey), overlaid on the autoencoder's prototype for its cluster. Above it are the class, how often a failure follows it, the likely cause, and a ⚠️ when the shape is unusual.
   - Right: the LightGBM risk drivers, as before.
4. **"How the amp-card classes are defined"**: an expandable explanation of the rules and the literature they come from.
5. The well's data table now includes the class and likely cause for each day.

**Changes to `src/esp_ampcard.py`**
- New `classify()` function for scoring new data. On the training data it gives the same classes as the training run 100% of the time.
- The saved model now also stores each cluster's shape name, its median card and the per-class failure rates.
- I retrained it. The class failure rates came out the same as before.

**Things to know**
- **First load takes about 12 seconds longer** while the amp cards are computed, then it's cached.
- **The app now needs a third model**, `model/esp_ampcard_ae.pt`. If you retrain from scratch, run `src/esp_lstm_ae.py`, then `src/esp_failure_clf.py`, then `src/esp_ampcard.py`.
- **The amp-card class isn't an input to the risk model yet**, so they're two separate readings shown side by side. Testing it as a LightGBM input is the next step in the notes.

I added these features to `docs/esp_failure_risk_notes.md`. Nothing is committed to git yet.

## You — 2026-11-02T13:21

add the amp card class as a LightGBM feature

## Claude — 2026-11-02T13:23

The summary lines didn't print, so something failed after the folds. Checking the error:

## Claude — 2026-11-02T13:23

My blanket `OUT /` → `out /` replacement also hit `ae.OUT`. Fixing that one reference:

## Claude — 2026-11-02T13:26

Now the app: reuse the amp cards that `score_wells` already computed, and give the new features readable names.

## Claude — 2026-11-02T13:26

The app runs with no errors across wells and dates. First load takes about 17 seconds because the amp cards are now computed once, inside the risk scoring. I'll record the ablation in the notes.

## Claude — 2026-11-02T13:26

The amp-card class is now a LightGBM input and part of the saved model the app uses. But **it doesn't measurably improve the predictions.** I compared cross-validation runs with and without it, using the same well-grouped folds:

| Alarm budget | Without amp card: caught / false alarms per well-year | With amp card |
|---|---|---|
| 1% of normal days | 11.4% / 0.30 | 10.5% / 0.29 |
| 2% | 13.3% / 0.40 | **16.2%** / 0.45 |
| 5% | 23.8% / **0.80** | 23.8% / 0.95 |

- **None of the differences is meaningful.** Resampling wells 300 times gives a PR-AUC change of +0.007 (95% interval −0.008 to +0.027) and an AUC change of −0.001 (−0.011 to +0.009). Both intervals include zero, and with 105 failures the swings in the table are a few events either way.
- **The model barely uses the new features.** The 27 amp-card features get 5% of the model's total importance; the most-used one is the card autoencoder's error, ranked 24th of 91.
- **The likely reason is that the model already has this information.** Shutdown counts and days since the last shutdown cover cycling and trips, and the current-vs-baseline deviations cover overload and underload. The amp card mostly re-expresses those in literature terms.

**What changed**
- `src/esp_failure_clf.py`: `ampcard_features()` builds the 27 new inputs:
  - today's class
  - days in the last 30 spent in cycling, recent trip, erratic, gradual rise, underload and speed change
  - the count of unusual cards and the card autoencoder's error
  - the likely-cause flags
- Two new options: `--no_ampcard` to switch the features off, and `--tag name` to write a comparison run without overwriting the saved models.
- `model/esp_failure_lgbm.*` was retrained with the amp-card features: 91 features, threshold 0.634, the JSON now records `uses_ampcard: true`, and the app reports 23% caught and 0.91 false alarms per well-year.
- **App:** the amp cards are now computed once inside the risk scoring and reused by the amp-card panels, and the drivers chart labels the new features, e.g. "Amp card 'recent trip' days, last 30". I tested it across wells and dates with Streamlit's headless test runner; first load takes about 17 seconds, then it's cached.
- The ablation results are in `docs/esp_failure_risk_notes.md`, and the two comparison runs are in `output/failure_clf_base/` and `output/failure_clf_ampcard/`.

**Caveat:** the card autoencoder was trained once on all wells rather than per fold. It never sees the failure labels, so any optimism should be small.

Since it adds no measurable accuracy, either version is reasonable to keep. The amp-card version shows familiar labels in the drivers chart; the version without it is simpler and has slightly fewer false alarms at the 5% budget. To switch back, run `python src/esp_failure_clf.py --no_ampcard`; the app picks up the change automatically.

## You — 2026-11-02T13:26

save all the conversation

## Claude — 2026-11-02T13:27

The retrain order in the notes is now out of date: the amp-card model has to run before the failure model. I'll fix that, and the stale app note, files table and next-steps list.

## Claude — 2026-11-02T13:27

`docs/esp_failure_risk_notes.md` now covers the whole session (318 lines):
1. The amp-card research verdict and literature, with links
2. Data findings (why `AMPERAGE` isn't SCADA, the 20 A pump-off floor, failure labels)
3. How the pipeline works
4. Results of the 30-day model
5. Files
6. How to run and how to tune
7. Amp-card classification, plus the test of amp-card features in LightGBM (no meaningful gain)
8. Open questions and next steps

This round I corrected sections 5, 6 and 8. The retrain order now includes the amp-card step, which must run before the 30-day model:
```bash
python src/esp_lstm_ae.py
python src/esp_ampcard.py
python src/esp_failure_clf.py
streamlit run risk_app.py
```

For a word-for-word copy of the conversation, type `/export` in this session. The raw transcript is also at `~/.claude/projects/-home-rianr-pypro-myvenv-esp-forecasting-src/0838d459-6096-43d5-a9a2-f86057d95fdf.jsonl`.

Nothing is committed to git yet. I can commit the new code, models, outputs and `docs/` if you want; I'd leave out `src/datascratch.py`, which has changes I didn't make.

## You — 2026-11-02T13:29

where is the file for ampcard?

## Claude — 2026-11-02T13:29

The amp-card files are all in your project folder, `/home/rianr/pypro/myvenv/esp_forecasting/`:

**Code**
- `src/esp_ampcard.py`: builds the cards, trains the autoencoder, assigns the classes and provides `classify()`. Run it with `python src/esp_ampcard.py`.

**Model**
- `model/esp_ampcard_ae.pt`: the trained amp-card autoencoder, with its cluster names, prototype cards and per-class failure rates.

**Outputs** (in `output/ampcard/`)
- `prototypes.png`: the median card for each shape family. Look at this one first.
- `cards.csv`: one row per well per day with the class, shape, likely cause and unusual flag.
- `class_vs_failure.csv`: how often each class is followed by a failure within 30 days.
- `clusters.csv`: the autoencoder's shape clusters and how each one was named.
- `summary.json`: card counts and settings.

**Where it's used**
- `risk_app.py`: the amp-card column, class panel and selected-day card in the app.
- `src/esp_failure_clf.py`: turns the classes into LightGBM features (`ampcard_features()`).
- `docs/esp_failure_risk_notes.md`, section 7: the explanation and results.

