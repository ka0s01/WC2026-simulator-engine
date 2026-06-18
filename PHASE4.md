# Phase 4 — Match Prediction Model

**Output:** `models/home_goals_model.json`, `models/away_goals_model.json` — two XGBoost regressors predicting home and away goals given team Elo ratings.

---

## Original Plan

The original plan was a three-part pipeline:

1. **Build Elo ratings** from `results.csv` — replay all 45k historical matches chronologically, maintain a running Elo dict, snapshot each team's Elo before every match. Save to `elo_history.csv`.

2. **Extract real xG from StatsBomb** — for every WC match in StatsBomb (1958–2018) plus major tournaments (Euro 2020, Copa America 2024, AFCON 2023), sum shot-level xG per team per match. Use this as the training target instead of raw goals — cleaner signal since xG removes luck.

3. **Train XGBoost** on the joined dataset (Elo features + StatsBomb xG targets). Validate on WC 2022 as a completely held-out tournament.

The reasoning: Elo gives you historical strength for every nation across 45k matches. StatsBomb xG gives you a better target variable than raw goals. Combining both was "Option C" — best of both worlds.

---

## Temporal Cutoffs (Critical)

```python
TRAIN_CUTOFF = "2022-11-19"  # everything before WC 2022 goes into training
VAL_START    = "2022-11-20"  # WC 2022 start date
VAL_END      = "2022-12-18"  # WC 2022 final
```

- `results.csv` — train on everything before Nov 20 2022. WC 2022 window is validation only. Everything after Dec 18 2022 (including ongoing WC 2026 matches) is blocked entirely.
- `statsbomb/` — WC 2022 (competition 43, season_id 106) is validation only, never touched during training.
- `shootouts.csv` — same cutoff logic, WC 2022 shootouts are validation only.

---

## StatsBomb Data Inventory

StatsBomb contains far more than just WC 2018/2022. Full competition list explored via `competitions.json`. Relevant competitions:

| Competition ID | Name | Seasons Available |
|---|---|---|
| 43 | FIFA World Cup | 1958, 1962, 1970, 1974, 1986, 1990, 2018, **2022 (val only)** |
| 55 | UEFA Euro | 2020, 2024 |
| 223 | Copa America | 2024 |
| 1267 | African Cup of Nations | 2023 |

Season IDs (StatsBomb uses numeric IDs, not years):

```python
STATSBOMB_COMPS = {
    43  : [3, 55, 54, 51, 272, 270, 269],  # WC 2018, 1990, 1986, 1974, 1970, 1962, 1958
    55  : [282, 43],                         # Euro 2024, 2020
    223 : [282],                             # Copa America 2024
    1267: [107],                             # AFCON 2023
}
WC2022_SEASON_ID = 106  # validation only
```

---

## What Actually Happened — Major Deviations

### Deviation 1 — StatsBomb xG training set was too small

Extracted real xG from StatsBomb by summing `shot.statsbomb_xg` per team per match across all training competitions. After applying the temporal cutoff (dropping post-Nov 2022 matches — Euro 2024, Copa 2024, AFCON 2023 all fell outside), the usable training set was:

- WC 1958–2018: ~82 matches
- UEFA Euro 2020: 51 matches
- **Total: 133 matches**

Trained XGBoost on these 133 rows with xG as the target. Cross-validation results:

```
Home xG CV MAE: 1.07 (+/- 0.15)
Away xG CV MAE: 0.95 (+/- 0.11)
```

Training MAE was 0.47/0.35 — a huge gap vs CV MAE indicating severe overfitting. 133 samples is simply not enough for XGBoost to generalise.

Tried other models on the same 133 rows:

| Model | Home CV MAE | Away CV MAE |
|---|---|---|
| Ridge | 0.9336 | 0.8154 |
| Lasso | 0.9330 | 0.8149 |
| Random Forest | 1.0526 | 0.9477 |
| Gradient Boost | 1.1688 | 0.9864 |
| XGBoost | 1.0708 | 0.9519 |

Ridge performed best but still weak. Validated Ridge on WC 2022:
- Winner accuracy: **48.4%** — worse than random in context.

### Deviation 2 — Switched to goals-based training on full results.csv

Abandoned StatsBomb xG as the training target. Switched to `home_score`/`away_score` from `results.csv` as the target variable. This gave 45,686 training rows instead of 133.

Additional changes:
- Dropped all friendly matches (`tournament == 'Friendly'`) — 17,444 rows removed, leaving 28,288 competitive matches
- Added recency weighting — exponential decay `exp(0.05 * (year - max_year))` so recent matches count more
- Added WC match bonus — WC matches get 2× sample weight on top of recency weight

Final training set: **28,288 competitive international matches, 1872–Nov 2022**.

### Deviation 3 — Team name mismatches between StatsBomb and results.csv

StatsBomb and `results.csv` use different team names. Required a name map:

```python
TEAM_NAME_MAP = {
    "Côte d'Ivoire"     : "Ivory Coast",
    "Cape Verde Islands": "Cape Verde",
    "Congo DR"          : "DR Congo",
}
```

Also discovered home/away assignment is arbitrary at neutral venues — StatsBomb and `results.csv` often flip home/away for the same WC match. Fixed by creating an order-independent `match_key`:

```python
match_key = '_'.join(sorted([home_team, away_team]))
```

---

## Elo Rating System

Built from scratch by replaying all matches in `results.csv` chronologically up to `TRAIN_CUTOFF`.

```python
K = 32
DEFAULT_ELO = 1500

expected_home = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
actual_home   = 1 (win) / 0.5 (draw) / 0 (loss)
elo[home]     = home_elo + K * (actual_home - expected_home)
elo[away]     = away_elo + K * ((1 - actual_home) - (1 - expected_home))
```

Final Elo ratings at TRAIN_CUTOFF (Nov 19 2022):

| Team | Elo |
|---|---|
| Brazil | 2081.7 |
| Argentina | 2041.0 |
| Spain | 1984.4 |
| France | 1961.2 |
| England | 1935.7 |
| Germany | 1926.4 |

Saved to `data/processed/elo_history.csv` — one row per historical match with pre-match Elo snapshot for both teams.

---

## Final Model

**Architecture:** Two independent XGBoost regressors — one for home goals, one for away goals.

**Features:**
```python
features = ['home_elo', 'away_elo', 'elo_diff', 'is_worldcup']
```

**Hyperparameters:**
```python
XGBRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.5,
    random_state=42
)
```

**Training data:** 28,288 competitive international matches (no friendlies), with recency + WC sample weights.

**Target:** `home_score`, `away_score` (actual goals, not xG).

---

## Validation Results — WC 2022

For validation, each WC 2022 team's Elo was taken from the final state of `elo_history` (i.e. their rating as of Nov 19 2022, the day before the tournament started).

| Metric | Value |
|---|---|
| Winner prediction accuracy | **53.1%** |
| Home goals MAE | 1.04 |
| Away goals MAE | 0.84 |

**Sanity check — model predictions make intuitive sense:**

| Matchup | Pred Home | Pred Away | Actual |
|---|---|---|---|
| Brazil vs weak team (WC) | 5.1 | 0.6 | ✓ direction |
| Brazil vs France (WC) | 1.59 | 1.53 | ✓ very close |
| Equal teams (WC) | 1.29 | 1.11 | ✓ ~2.4 total goals |
| France vs Australia | 1.95 | 0.80 | Actual 4-1, direction ✓ |
| Argentina vs Saudi Arabia | 2.40 | 0.73 | Actual 1-2, upset ✗ |
| Spain vs Costa Rica | 2.12 | 0.72 | Actual 7-0, direction ✓ |

53.1% winner accuracy is below ideal. Context: betting models with far richer features (injuries, form, tactical data) typically achieve 55–60% on World Cup matches. We're at 53% with Elo only. The gap will be addressed at inference time by using `team_vectors.csv` (FC26 ratings + Understat xG) to produce a synthetic Elo-equivalent per team for WC 2026 rather than relying on raw historical Elo.

---

## How Inference Works for WC 2026

The model was trained on Elo features. At inference time for WC 2026, we don't use raw historical Elo — we convert `team_vectors.csv` into a synthetic strength score in Elo units:

```python
def team_vector_to_elo(row):
    score = (
        0.4 * row['top11_overall'] +
        0.2 * row['atk_overall'] +
        0.2 * row['def_overall'] +
        0.1 * row['mid_overall'] +
        0.1 * row['gk_overall']
    )
    # scale from FC26 range (~70-90) to Elo range (~1400-2100)
    return scale(score, in_min=70, in_max=90, out_min=1400, out_max=2100)
```

Teams with Understat xG data get a form adjustment nudge on top of this base score.

---

## Where StatsBomb Goes From Here

StatsBomb xG data was not used in the final trained model but is not wasted:

- **Phase 5 (Dixon-Coles calibration)** — WC 2018 shot-level xG values are ideal for calibrating the score distribution. The model predicts expected goals; Dixon-Coles converts those into scoreline probabilities. Calibrating against real WC xG ensures draw rate hits 25–28% and avg goals/game hits 2.4–2.6.
- **Phase 5 validation** — WC 2022 StatsBomb match data used to cross-check predicted vs actual scoreline distributions.

---

## Files Produced in Phase 4

```
data/processed/
├── elo_history.csv         ← pre-match Elo snapshot for every historical match
├── training_set.csv        ← 28,288 competitive matches with Elo features + goals
└── val_set.csv             ← 64 WC 2022 matches with predictions

models/
├── home_goals_model.json   ← XGBoost home goals predictor
└── away_goals_model.json   ← XGBoost away goals predictor

data/processed/
└── wc2022_validation.png   ← validation plots (scatter, winner accuracy, per-match bar chart)
```

## Scripts / Notebooks

```
notebooks/
└── 04_match_model.ipynb    ← full Phase 4 pipeline
```

---

## Phase 5 Plan (Next Session)

Dixon-Coles score sampler calibration.

- Load WC 2018 StatsBomb xG values (real shot data)
- Use predicted goals from Phase 4 model as lambda values for Poisson
- Add Dixon-Coles rho correction parameter to fix underestimation of draws and low-scoring games
- Calibrate rho until: draw rate = 25–28%, avg goals/game = 2.4–2.6
- Output: `dixon_coles_rho` constant used in Phase 6 Monte Carlo engine