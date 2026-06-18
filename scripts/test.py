import pandas as pd
import unicodedata
from rapidfuzz import process, fuzz

def norm(s):
    s = str(s).lower().strip()
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

players = pd.read_csv("data/processed/players_master.csv")
understat = pd.read_csv("data/raw/understat_player_stats.csv")

understat = understat[understat['league'] != 'RUS-Premier League']
understat = understat[understat['time'] >= 200]

players['name_norm2'] = players['name'].apply(norm)
understat['name_norm2'] = understat['player_name'].apply(norm)

# exact after norm
und_norm_list = understat['name_norm2'].tolist()
und_norm_set = set(und_norm_list)

exact_matched = players[players['name_norm2'].isin(und_norm_set)]
unmatched = players[~players['name_norm2'].isin(und_norm_set)]

print(f"Exact after norm: {len(exact_matched)}")
print(f"Still unmatched: {len(unmatched)}")

# fuzzy on the unmatched only
results = []
for _, row in unmatched.iterrows():
    match = process.extractOne(row['name_norm2'], und_norm_list, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= 85:
        results.append({
            'squad_name': row['name'],
            'und_match': understat.iloc[match[2]]['player_name'],
            'score': match[1],
            'team': row['team']
        })

fuzzy_df = pd.DataFrame(results)
print(f"\nFuzzy matches (>=85): {len(fuzzy_df)}")
print("\nSample — check these for correctness:")
print(fuzzy_df.head(30).to_string())