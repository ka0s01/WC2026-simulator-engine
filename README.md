# WC 2026 Simulator Engine

A data pipeline and Monte Carlo simulator for the 2026 FIFA World Cup. Runs 10,000 complete tournament simulations and outputs probability distributions for all 48 teams, connected to a Next.js frontend via a single JSON file.

---

## How It Works

The simulator answers one question: given everything we know about these 48 squads going into the tournament, how likely is each team to win?

The answer comes from running the tournament 10,000 times and counting outcomes. To do that, you need to be able to simulate individual matches. To simulate matches, you need a model that knows how good each team is. And to know how good each team is, you need data. So the work, roughly, was: get the data → build team quality scores → train a match model → simulate the tournament.

---

## Data

The starting point was assembling several sources:

- **Kaggle international results** (45,000+ matches, 1872–present) — the backbone for training the match model. Every competitive international result with scores.
- **Wikipedia WC 2026 squads** — scraped to get the confirmed 26-man roster for all 48 teams (1,248 players). This had to be scraped fresh since the tournament was already underway.
- **FC26 player ratings** — EA's football ratings for ~18k players. Used as quality priors: pace, shooting, passing, defending, dribbling, physicality, overall.
- **Understat** — player xG and xA from the current club season, covering EPL, La Liga, Bundesliga, Serie A, Ligue 1, and Russian Premier League. This is the "current form" signal.
- **StatsBomb Open Data** — shot-level match data for WC 2018, WC 2022, Euro 2020/2024, Copa America 2024, AFCON 2023. Used for validation, not training.

Two sources that were originally planned didn't work out: FBref blocks scraping behind Cloudflare, and Sofascore moved to Chromium fingerprinting. Understat was used as the replacement for FBref. It doesn't cover non-EU leagues, so players at Saudi, Brazilian, Turkish, and other clubs outside the six covered leagues have no form data — they fall back to FC26 ratings only.

---

## Player Identity Resolution

Getting the right FC26 rating for each of the 1,248 squad players was harder than expected. Name matching alone produced bad results: Vinícius Júnior was matched to a 74-rated player instead of his actual 89, two Dutch players with similar names were swapped, and five Egyptian players the model didn't recognise were all matched to Mohamed Salah.

The fix was to use **date of birth + nationality** as the primary join key rather than names. FC26 includes DOB for every player, and a player born on a specific date with a specific nationality is almost always unique. That resolved 824 of the 1,248 players cleanly.

For ambiguous cases (same DOB + nationality, multiple FC26 entries), and for players who simply didn't appear in the DOB lookup, a local Mistral 7B model via Ollama was used as a fallback — given the squad player's name, club, and a ranked list of FC26 candidates, it picked the right one. After LLM resolution, total FC26 coverage went from 827 to 1,040 out of 1,248 players. The remaining 203 (unlicensed players like Neymar, Courtois, Lukaku, plus genuinely obscure players) were filled with positional median values.

The Understat matching used the same LLM approach: filter by club name, send top 5 name candidates to Mistral, confirm or reject.

Output: `data/processed/player_id_map.csv` — one row per squad player with confirmed FC26 and Understat identities.

---

## Team Feature Vectors

With clean player-level data, each team was compressed into a single row of 17 features. Positional groups (attackers, midfielders, defenders, goalkeepers) were averaged separately and weighted by FC26 overall rating, so better players count more when computing a group's quality score.

| Feature | What it captures |
|---|---|
| `atk_overall`, `atk_shooting`, `atk_pace`, `atk_dribbling` | Attacking quality |
| `mid_overall`, `mid_passing`, `mid_physic` | Midfield quality |
| `def_overall`, `def_defending`, `def_physic` | Defensive quality |
| `gk_overall` | Goalkeeping quality |
| `top11_overall` | Best 11 players by rating |
| `squad_depth` | Gap between starters and bench |
| `avg_caps` | Tournament experience |
| `atk_xG_per90`, `atk_xA_per90` | Current attacking form (Understat, EU players only) |

xG/xA is only populated if at least 40% of a team's attackers have Understat data — below that threshold it's too sparse to be meaningful. That threshold is met by 17 of the 48 teams.

The top-5 teams by `top11_overall` came out as France (86.73), Spain (86.73), Brazil (85.91), Germany (85.27), Portugal (85.27) — which matches expectations well.

Output: `data/processed/team_vectors.csv`

---

## Match Prediction Model

The match model takes two team strength scores and predicts how many goals each team scores. It's trained on competitive international matches from `results.csv` — friendlies dropped, leaving 28,288 matches from 1872 to November 2022.

Team strength is represented as **Elo ratings** built by replaying all those matches chronologically (K=32, starting at 1500). Every match in training data has a pre-match Elo snapshot for both teams, and that's what the model sees.

Originally the plan was to use StatsBomb xG values as the training target instead of actual goals — xG is a cleaner signal because it removes finishing luck. That approach was tested and failed: after applying strict temporal cutoffs (WC 2022 can't touch training data), only 133 StatsBomb rows were usable. Every model overfit badly on 133 rows. Switched back to raw goals from `results.csv`, which gave 28,288 rows and worked.

**Two independent XGBoost regressors** — one for home goals, one for away goals. Features are `home_elo`, `away_elo`, `elo_diff`, `is_worldcup`. Training samples are weighted by recency (exponential decay) and a 2× bonus for WC matches.

Validated on WC 2022 as a completely held-out tournament — none of those 64 matches touched training:

| Metric | Value |
|---|---|
| Winner prediction accuracy | 53.1% |
| Home goals MAE | 1.04 |
| Away goals MAE | 0.84 |

At inference time for WC 2026, historical Elo isn't used directly. Instead, each team's `team_vectors.csv` features are converted to a synthetic Elo-equivalent score (weighted combination of positional FC26 ratings, scaled to Elo range), with a small nudge for teams that have Understat xG data. This means the model benefits from current squad quality rather than just historical win/loss record.

Models: `models/home_goals_model.json`, `models/away_goals_model.json`

---

## Score Distribution

The match model outputs expected goals (lambdas). To simulate a scoreline, you draw from a Poisson distribution — but raw Poisson underestimates draws and low-scoring results like 0-0 and 1-1, which are common in international football.

Dixon-Coles fixes this by adding a correction factor (rho) to the four low-scoring scorelines only:

```
P(0-0) is boosted, P(1-1) is boosted, P(1-0) and P(0-1) are adjusted accordingly
All other scorelines: unchanged
```

Rho was calibrated by grid search on 500 subsampled pre-2018 WC matches, targeting a draw rate of 25–28%. The chosen value is **rho = -0.29**, which produces a draw rate of ~25.2% on calibration data and ~28% on the WC 2022 held-out set.

Saved to: `models/dixon_coles_params.json`

---

## Monte Carlo Simulation

With a calibrated score sampler and a model that can predict goals for any two teams, simulating the full tournament is straightforward:

1. Simulate all 48 group stage matches, build the tables, advance the top 2 from each group plus the 8 best third-place teams
2. Run the knockout bracket (R32 → R16 → QF → SF → Final) using the same match model
3. Draws in knockout rounds go to extra time (lambdas × 0.75) then penalties, using historical shootout records from `shootouts.csv` per nation
4. Record who wins each round

Run that 10,000 times. The win probability for each team is how often they won. The final output is `data/processed/simulation_results.json`, consumed directly by the frontend.

---

## Frontend Connection

The pipeline emits one file. Copy it to the frontend repo and redeploy — that's the entire integration:

```bash
cp data/processed/simulation_results.json ../wc2026-web/data/simulation_results.json
```

The frontend reads everything through `lib/teams.ts`. See [CONNECTION.md](CONNECTION.md) for the JSON schema, slug conventions, and a validation checklist to run before copying.

---

## Project Structure

```
WC2026-simulator-engine/
├── scripts/
│   ├── 01_download_static.py       ← clone StatsBomb + OpenFootball
│   ├── 02_scrape_squads.py         ← Wikipedia squad scraper
│   ├── 03_scrape_form.py           ← Understat POST API
│   ├── 04_build_player_matcher.py  ← players_master.csv
│   └── 05_resolve_player_names.py  ← DOB + nationality + LLM identity resolution
├── notebooks/
│   ├── 03_team_vectors_final.ipynb
│   ├── 04_match_model.ipynb
│   ├── 05_dixon_coles.ipynb
│   └── 06_monte_carlo_simulation.ipynb
├── models/
│   ├── home_goals_model.json
│   ├── away_goals_model.json
│   └── dixon_coles_params.json
├── data/
│   ├── raw/
│   │   ├── statsbomb/
│   │   ├── openfootball/
│   │   ├── kaggle/
│   │   ├── squads.csv
│   │   └── understat_player_stats.csv
│   └── processed/
│       ├── player_id_map.csv
│       ├── players_master.csv
│       ├── team_vectors.csv
│       ├── elo_history.csv
│       └── simulation_results.json
└── CONNECTION.md                   ← frontend wiring guide
```

---

## Setup

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

To regenerate from scratch, run the scripts in order, then the notebooks (03 → 04 → 05 → 06).

---

## Known Limitations

- **53.1% winner accuracy** — the gap to betting-market models (~58%) is real. Those models have injury data, tactical signals, and squad selection — this one has Elo and FC26 ratings.
- **EU form data only** — 807 of 1,248 players have no Understat coverage. Teams like Saudi Arabia, Iran, New Zealand, and South Africa are predicted almost entirely from FC26 ratings. The model is less confident about them, which is honest.
- **17/48 teams have xG features** — below the 40% attacker coverage threshold, xG/xA are null and the model falls back to FC26-only for that team.
- **203 players on positional fill** — EA licensing exclusions plus obscure players. Neymar, Courtois, Lukaku all fall into this group.

---

## What's Left

- **Live results integration** — WC 2026 is underway as of June 19 2026. Re-running the Monte Carlo only (no retraining) with actual group stage results fed in would sharpen remaining probabilities significantly.
- **Frontend deployment** — run the CONNECTION.md validation checklist and copy the JSON.
- **Non-EU form coverage** — a paid FBref proxy or FootyStats would close the Understat gap for Saudi, Brazilian, and Turkish club players.
- **Penalty model** — `shootouts.csv` has per-nation shootout history. Currently used as flat rates; a Bayesian model per team would be more precise.
