import pandas as pd
import unicodedata
import re
from rapidfuzz import process, fuzz
from collections import defaultdict

fc26 = pd.read_csv("data/raw/kaggle/FC_26.csv", low_memory=False)
squads = pd.read_csv("data/raw/squads.csv")

def normalize(s):
    if not isinstance(s, str):
        return ""
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    s = re.sub(r"[^\x00-\x7F]+", "", s)
    return s.strip().lower()

NATION_MAP = {
    "South Korea":       "Korea Republic",
    "USA":               "United States",
    "IR Iran":           "Iran",
    "Ivory Coast":       "Cote d'Ivoire",
    "Cape Verde":        "Cabo Verde",
    "Congo DR":          "Congo DR",
    "Trinidad & Tobago": "Trinidad and Tobago",
    "Turkey":            "Turkiye",
}

squads["nationality_mapped"] = squads["team"].replace(NATION_MAP)
fc26["long_norm"]  = fc26["long_name"].apply(normalize)
fc26["short_norm"] = fc26["short_name"].apply(normalize)
fc26["nat_norm"]   = fc26["nationality_name"].apply(normalize)
squads["name_norm"] = squads["name"].apply(normalize)
squads["nat_norm"]  = squads["nationality_mapped"].apply(normalize)

FM_COLS = ["long_name", "short_name", "nationality_name", "overall", "pace",
           "shooting", "passing", "dribbling", "defending", "physic", "player_positions"]

# Build nationality-scoped pools
nat_long_pool  = defaultdict(list)
nat_short_pool = defaultdict(list)
for idx, row in fc26.iterrows():
    nat = row["nat_norm"]
    nat_long_pool[nat].append((row["long_norm"], idx))
    nat_short_pool[nat].append((row["short_norm"], idx))

all_long  = list(fc26["long_norm"])
all_short = list(fc26["short_norm"])

results = []
for _, row in squads.iterrows():
    norm = row["name_norm"]
    nat  = row["nat_norm"]

    # 1. Exact long name
    exact = fc26[fc26["long_norm"] == norm]
    if not exact.empty:
        idx = exact.index[0]
        results.append({**row, **fc26.loc[idx, FM_COLS], "match_type": "exact_long", "match_score": 100})
        continue

    # 2. Exact short name
    exact = fc26[fc26["short_norm"] == norm]
    if not exact.empty:
        idx = exact.index[0]
        results.append({**row, **fc26.loc[idx, FM_COLS], "match_type": "exact_short", "match_score": 100})
        continue

    # 3. Fuzzy short within nationality — threshold 75 (safe because nat guards it)
    pool = nat_short_pool.get(nat, [])
    if pool:
        names = [p[0] for p in pool]
        idxs  = [p[1] for p in pool]
        match, score, pos = process.extractOne(norm, names, scorer=fuzz.token_sort_ratio)
        if score >= 75:
            idx = idxs[pos]
            results.append({**row, **fc26.loc[idx, FM_COLS], "match_type": "fuzzy_nat_short", "match_score": score})
            continue

    # 4. Fuzzy long within nationality — threshold 75
    pool = nat_long_pool.get(nat, [])
    if pool:
        names = [p[0] for p in pool]
        idxs  = [p[1] for p in pool]
        match, score, pos = process.extractOne(norm, names, scorer=fuzz.token_sort_ratio)
        if score >= 75:
            idx = idxs[pos]
            results.append({**row, **fc26.loc[idx, FM_COLS], "match_type": "fuzzy_nat_long", "match_score": score})
            continue

    # 5. Global fallback — high threshold, no nat guard
    match, score, pos = process.extractOne(norm, all_short, scorer=fuzz.token_sort_ratio)
    if score >= 90:
        results.append({**row, **fc26.loc[pos, FM_COLS], "match_type": "fuzzy_global", "match_score": score})
    else:
        results.append({**row, "match_type": "unmatched", "match_score": score,
                        **{c: None for c in FM_COLS}})

merged = pd.DataFrame(results)

print("=== MATCH BREAKDOWN ===")
print(merged["match_type"].value_counts())
print(f"\nCoverage: {merged['overall'].notna().sum()} / {len(merged)} ({merged['overall'].notna().mean()*100:.1f}%)")

# Sanity check
check = ["Son Heung-min", "Kylian Mbappé", "Erling Haaland", "Vinícius Júnior",
         "Lionel Messi", "Thibaut Courtois", "Romelu Lukaku", "Moisés Caicedo"]
print("\n=== BIG NAME CHECK ===")
for name in check:
    r = merged[merged["name"] == name]
    if not r.empty:
        r = r.iloc[0]
        print(f"  {name:25s} → {r['match_type']:20s} | FC26: {str(r.get('long_name','?'))[:35]:35s} | overall: {r.get('overall','?')}")

# False positive check — show all fuzzy matches with score 75-79 for manual review
print("\n=== FUZZY MATCHES 75-79 (verify no false positives) ===")
sus = merged[
    merged["match_type"].isin(["fuzzy_nat_short", "fuzzy_nat_long"]) &
    (merged["match_score"] < 80)
][["name", "team", "long_name", "nationality_name", "match_score", "overall"]]
print(sus.to_string(index=False))

print(f"\nUnmatched: {merged['match_type'].eq('unmatched').sum()}")
merged.to_csv("data/processed/players_master.csv", index=False)
print("Saved → data/processed/players_master.csv")