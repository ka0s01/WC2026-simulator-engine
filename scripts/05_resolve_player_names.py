import pandas as pd
import unicodedata
import json
import ollama
from rapidfuzz import fuzz
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
SQUADS     = ROOT / "data/raw/squads.csv"
FC26       = ROOT / "data/raw/kaggle/FC_26.csv"
UNDERSTAT  = ROOT / "data/raw/understat_player_stats.csv"
OUTPUT     = ROOT / "data/processed/player_id_map.csv"

# ── helpers ────────────────────────────────────────────────────────────────
def norm(s):
    s = str(s).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def ask_ollama(prompt):
    response = ollama.chat(
        model='mistral:latest',
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content'].strip()

# ── load data ──────────────────────────────────────────────────────────────
print("Loading data...")
squads    = pd.read_csv(SQUADS)
fc26      = pd.read_csv(FC26, low_memory=False)
understat = pd.read_csv(UNDERSTAT)

# clean understat
understat = understat[understat['league'] != 'RUS-Premier League']
understat = understat[understat['time'] >= 200]
understat = understat.sort_values('time', ascending=False).drop_duplicates(subset='player_name', keep='first')
understat['xG_per90'] = understat['xG'] / (understat['time'] / 90)
understat['xA_per90'] = understat['xA'] / (understat['time'] / 90)

# nation map squads → fc26
nation_map = {
    'Cape Verde'     : 'Cape Verde Islands',
    'Curaçao'        : 'Curacao',
    'Czech Republic' : 'Czechia',
    'DR Congo'       : 'DR Congo',
    'Ivory Coast'    : "Côte d'Ivoire",
    'South Korea'    : 'Korea Republic',
    'Turkey'         : 'Türkiye',
}
squads['nationality_name'] = squads['team'].map(nation_map).fillna(squads['team'])
squads['dob'] = pd.to_datetime(squads['dob']).dt.strftime('%Y-%m-%d')
fc26['dob']   = pd.to_datetime(fc26['dob']).dt.strftime('%Y-%m-%d')

# ── STEP 1: DOB + nationality join ─────────────────────────────────────────
print("\nStep 1: DOB + nationality matching...")

merged = squads.merge(
    fc26[['long_name', 'short_name', 'dob', 'nationality_name', 'club_name', 'overall',
          'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic', 'player_positions']],
    on=['dob', 'nationality_name'],
    how='left'
)

match_counts  = merged.groupby('name')['long_name'].count()
unique_names  = match_counts[match_counts == 1].index
ambiguous_names = match_counts[match_counts > 1].index
unmatched_names = match_counts[match_counts == 0].index

clean_matches = merged[merged['name'].isin(unique_names)].copy()
ambiguous     = merged[merged['name'].isin(ambiguous_names)].copy()
unmatched     = squads[squads['name'].isin(unmatched_names)].copy()

print(f"  Clean matches : {len(clean_matches)}")
print(f"  Ambiguous     : {len(ambiguous['name'].unique())}")
print(f"  Unmatched     : {len(unmatched)}")

# ── STEP 2: LLM resolves ambiguous ────────────────────────────────────────
print("\nStep 2: LLM resolving ambiguous matches...")

ambiguous_resolved = []
for squad_name in ambiguous['name'].unique():
    squad_row  = squads[squads['name'] == squad_name].iloc[0]
    candidates = ambiguous[ambiguous['name'] == squad_name][['long_name', 'club_name', 'overall']].reset_index(drop=True)

    candidates_str = "\n".join([
        f"{i+1}. {row['long_name']} | club: {row['club_name']} | overall: {row['overall']}"
        for i, row in candidates.iterrows()
    ])

    prompt = f"""You are a football data expert. Match this squad player to the correct FC26 entry.

Squad player: {squad_row['name']} | club: {squad_row['club']} | DOB: {squad_row['dob']} | team: {squad_row['team']}

FC26 candidates (same nationality, same DOB):
{candidates_str}

Reply with ONLY the number of the correct candidate, or 0 if none match. No explanation."""

    reply = ask_ollama(prompt)

    try:
        pick = int(reply.strip()[0])
    except:
        pick = 0

    if pick > 0 and pick <= len(candidates):
        chosen = candidates.iloc[pick - 1]
        ambiguous_resolved.append({
            'name'             : squad_row['name'],
            'fc26_long_name'   : chosen['long_name'],
            'fc26_club'        : chosen['club_name'],
            'fc26_overall'     : chosen['overall'],
            'fc26_match_method': 'llm_ambiguous'
        })
        print(f"  ✓ {squad_row['name']} → {chosen['long_name']}")
    else:
        ambiguous_resolved.append({
            'name'             : squad_row['name'],
            'fc26_long_name'   : None,
            'fc26_club'        : None,
            'fc26_overall'     : None,
            'fc26_match_method': 'llm_ambiguous_no_match'
        })
        print(f"  ✗ {squad_row['name']} → no match")

ambiguous_resolved_df = pd.DataFrame(ambiguous_resolved)

# ── STEP 3: LLM fuzzy matches unmatched ───────────────────────────────────
print("\nStep 3: LLM fuzzy matching unmatched players...")

unmatched_resolved = []
for _, squad_row in unmatched.iterrows():
    # filter fc26 by nationality, get top 10 by name similarity
    nat_pool = fc26[fc26['nationality_name'] == squad_row['nationality_name']]
    
    if len(nat_pool) == 0:
        # try without nationality filter
        nat_pool = fc26.copy()

    # rank by name similarity and take top 10
    nat_pool = nat_pool.copy()
    nat_pool['name_score'] = nat_pool['long_name'].apply(
        lambda x: fuzz.token_sort_ratio(norm(squad_row['name']), norm(str(x)))
    )
    candidates = nat_pool.nlargest(10, 'name_score')[
        ['long_name', 'club_name', 'overall', 'dob', 'name_score']
    ].reset_index(drop=True)

    # skip if top candidate score is very low — likely not in FC26
    if candidates.iloc[0]['name_score'] < 50:
        unmatched_resolved.append({
            'name'             : squad_row['name'],
            'fc26_long_name'   : None,
            'fc26_club'        : None,
            'fc26_overall'     : None,
            'fc26_match_method': 'not_in_fc26'
        })
        continue

    candidates_str = "\n".join([
        f"{i+1}. {row['long_name']} | club: {row['club_name']} | DOB: {row['dob']} | overall: {row['overall']}"
        for i, row in candidates.iterrows()
    ])

    prompt = f"""You are a football data expert. Match this squad player to the correct FC26 entry.

Squad player: {squad_row['name']} | club: {squad_row['club']} | DOB: {squad_row['dob']} | team: {squad_row['team']}

FC26 candidates:
{candidates_str}

Reply with ONLY the number of the correct candidate, or 0 if none match. No explanation."""

    reply = ask_ollama(prompt)

    try:
        pick = int(reply.strip()[0])
    except:
        pick = 0

    if pick > 0 and pick <= len(candidates):
        chosen = candidates.iloc[pick - 1]
        unmatched_resolved.append({
            'name'             : squad_row['name'],
            'fc26_long_name'   : chosen['long_name'],
            'fc26_club'        : chosen['club_name'],
            'fc26_overall'     : chosen['overall'],
            'fc26_match_method': 'llm_unmatched'
        })
        print(f"  ✓ {squad_row['name']} → {chosen['long_name']}")
    else:
        unmatched_resolved.append({
            'name'             : squad_row['name'],
            'fc26_long_name'   : None,
            'fc26_club'        : None,
            'fc26_overall'     : None,
            'fc26_match_method': 'not_in_fc26'
        })
        print(f"  ✗ {squad_row['name']} → not in FC26")

unmatched_resolved_df = pd.DataFrame(unmatched_resolved)

# ── STEP 4: Understat matching for all 1248 ────────────────────────────────
print("\nStep 4: Understat matching...")

understat_resolved = []
for _, squad_row in squads.iterrows():
    # filter understat by club
    club_norm = norm(squad_row['club'])
    understat['club_norm'] = understat['team_title'].apply(norm)
    
    club_pool = understat[understat['club_norm'].apply(
        lambda x: fuzz.token_sort_ratio(x, club_norm) > 80
    )]

    if len(club_pool) == 0:
        understat_resolved.append({
            'name'              : squad_row['name'],
            'understat_name'    : None,
            'understat_xG_per90': None,
            'understat_xA_per90': None,
            'understat_mins'    : None,
        })
        continue

    # rank by name similarity
    club_pool = club_pool.copy()
    club_pool['name_score'] = club_pool['player_name'].apply(
        lambda x: fuzz.token_sort_ratio(norm(squad_row['name']), norm(str(x)))
    )
    candidates = club_pool.nlargest(5, 'name_score').reset_index(drop=True)

    # skip if top score too low
    if candidates.iloc[0]['name_score'] < 60:
        understat_resolved.append({
            'name'              : squad_row['name'],
            'understat_name'    : None,
            'understat_xG_per90': None,
            'understat_xA_per90': None,
            'understat_mins'    : None,
        })
        continue

    candidates_str = "\n".join([
        f"{i+1}. {row['player_name']} | club: {row['team_title']} | mins: {row['time']} | xG: {row['xG']:.2f}"
        for i, row in candidates.iterrows()
    ])

    prompt = f"""You are a football data expert. Match this squad player to the correct Understat entry.

Squad player: {squad_row['name']} | club: {squad_row['club']} | team: {squad_row['team']}

Understat candidates (same club):
{candidates_str}

Reply with ONLY the number of the correct candidate, or 0 if none match. No explanation."""

    reply = ask_ollama(prompt)

    try:
        pick = int(reply.strip()[0])
    except:
        pick = 0

    if pick > 0 and pick <= len(candidates):
        chosen = candidates.iloc[pick - 1]
        understat_resolved.append({
            'name'              : squad_row['name'],
            'understat_name'    : chosen['player_name'],
            'understat_xG_per90': chosen['xG_per90'],
            'understat_xA_per90': chosen['xA_per90'],
            'understat_mins'    : chosen['time'],
        })
        print(f"  ✓ {squad_row['name']} → {chosen['player_name']}")
    else:
        understat_resolved.append({
            'name'              : squad_row['name'],
            'understat_name'    : None,
            'understat_xG_per90': None,
            'understat_xA_per90': None,
            'understat_mins'    : None,
        })

understat_resolved_df = pd.DataFrame(understat_resolved)

# ── STEP 5: assemble player_id_map ────────────────────────────────────────
print("\nStep 5: Assembling player_id_map...")

# build fc26 part from clean matches
clean_fc26 = clean_matches[['name', 'long_name', 'club_name', 'overall']].copy()
clean_fc26.columns = ['name', 'fc26_long_name', 'fc26_club', 'fc26_overall']
clean_fc26['fc26_match_method'] = 'dob_nationality'

# combine all fc26 matches
fc26_all = pd.concat([
    clean_fc26,
    ambiguous_resolved_df[['name', 'fc26_long_name', 'fc26_club', 'fc26_overall', 'fc26_match_method']],
    unmatched_resolved_df[['name', 'fc26_long_name', 'fc26_club', 'fc26_overall', 'fc26_match_method']],
], ignore_index=True)

# join everything to squads
player_id_map = squads[['name', 'team', 'dob', 'club', 'position', 'caps', 'goals']].merge(
    fc26_all, on='name', how='left'
).merge(
    understat_resolved_df, on='name', how='left'
)

player_id_map.to_csv(OUTPUT, index=False)

print(f"\n✓ player_id_map.csv written — {len(player_id_map)} rows")
print(f"  FC26 matched       : {player_id_map['fc26_long_name'].notna().sum()}")
print(f"  Understat matched  : {player_id_map['understat_name'].notna().sum()}")
print(f"  FC26 match methods :\n{player_id_map['fc26_match_method'].value_counts()}")