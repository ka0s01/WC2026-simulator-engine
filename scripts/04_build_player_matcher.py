import pandas as pd
import numpy as np

# ── paths ──────────────────────────────────────────────────────────────────
SQUADS      = "data/raw/squads.csv"
FC26        = "data/raw/kaggle/FC_26.csv"
ID_MAP      = "data/processed/player_id_map.csv"
OUTPUT      = "data/processed/players_master.csv"

# ── load ───────────────────────────────────────────────────────────────────
print("Loading data...")
squads  = pd.read_csv(SQUADS)
fc26    = pd.read_csv(FC26, low_memory=False)
id_map  = pd.read_csv(ID_MAP)

# ── get full FC26 attributes via confirmed long_name ──────────────────────
fc26_attrs = fc26[[
    'long_name', 'overall', 'pace', 'shooting', 'passing',
    'dribbling', 'defending', 'physic', 'player_positions'
]].drop_duplicates(subset='long_name', keep='first')

# ── join id_map to fc26 attrs ─────────────────────────────────────────────
players = id_map.merge(
    fc26_attrs,
    left_on='fc26_long_name',
    right_on='long_name',
    how='left'
).drop(columns=['long_name'])

# ── broad_pos from position column ────────────────────────────────────────
pos_map = {
    'GK' : 'GK',
    'DF' : 'DEF',
    'MF' : 'MID',
    'FW' : 'ATT',
}
players['broad_pos'] = players['position'].map(pos_map)

# ── positional fill for unmatched players ─────────────────────────────────
# compute medians from matched players only
matched = players[players['fc26_long_name'].notna()]
fill_overall = matched.groupby('broad_pos')['overall'].median()
fill_attrs   = matched.groupby('broad_pos')[
    ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']
].median()

print("\nPositional fill values (overall):")
print(fill_overall)

def fill_row(row):
    if pd.notna(row['fc26_long_name']):
        return row
    pos  = row['broad_pos']
    row['overall']   = fill_overall.get(pos, 75)
    row['match_type'] = 'positional_fill'
    for col in ['pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic']:
        if pd.isna(row[col]):
            row[col] = fill_attrs.loc[pos, col] if pos in fill_attrs.index else 70
    return row

players = players.apply(fill_row, axis=1)

# ── final checks ──────────────────────────────────────────────────────────
print("\n=== MATCH BREAKDOWN ===")
print(players['fc26_match_method'].value_counts())
print(f"\nTotal players : {len(players)}")
print(f"FC26 matched  : {players['fc26_long_name'].notna().sum()}")
print(f"Positional fill: {(players['fc26_match_method'] == 'positional_fill').sum()}")
print(f"\nNull check (should all be 0):")
print(players[['overall','pace','shooting','passing','dribbling','defending','physic']].isnull().sum())
print(f"\nBroad pos counts:")
print(players['broad_pos'].value_counts())

# ── teams with fewest real matches ────────────────────────────────────────
team_coverage = players.groupby('team').apply(
    lambda g: (g['fc26_match_method'] != 'positional_fill').sum()
).sort_values()
print("\n=== TEAMS WITH FEWEST FC26 MATCHES ===")
print(team_coverage.head(10))

# ── save ──────────────────────────────────────────────────────────────────
players.to_csv(OUTPUT, index=False)
print(f"\n✓ players_master.csv saved → {OUTPUT}")