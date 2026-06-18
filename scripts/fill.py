import pandas as pd

merged = pd.read_csv("data/processed/players_master.csv")

print("=== CURRENT STATE ===")
print(f"Total players: {len(merged)}")
print(f"Matched: {merged['overall'].notna().sum()}")
print(f"Unmatched (need fill): {merged['overall'].isna().sum()}")

# Position cleanup — FC26 positions are messy, normalize to GK/DEF/MID/ATT
def broad_position(pos_str, squad_pos):
    """Classify to broad role. Use squad position as primary signal."""
    if not isinstance(squad_pos, str):
        squad_pos = ""
    p = squad_pos.upper().strip()
    if p == "GK":
        return "GK"
    elif p in ["CB", "RB", "LB", "RWB", "LWB", "DF"]:
        return "DEF"
    elif p in ["CM", "CDM", "CAM", "LM", "RM", "MF"]:
        return "MID"
    elif p in ["ST", "CF", "LW", "RW", "FW"]:
        return "ATT"
    # Fallback to FC26 position string
    if not isinstance(pos_str, str):
        return "MID"  # default
    pos_str = pos_str.upper()
    if "GK" in pos_str:
        return "GK"
    elif any(x in pos_str for x in ["CB", "RB", "LB", "WB"]):
        return "DEF"
    elif any(x in pos_str for x in ["CAM", "CDM", "CM", "LM", "RM"]):
        return "MID"
    elif any(x in pos_str for x in ["ST", "CF", "LW", "RW"]):
        return "ATT"
    return "MID"

merged["broad_pos"] = merged.apply(
    lambda r: broad_position(r.get("player_positions"), r.get("position")), axis=1
)

# Compute fill values — median overall by broad position across ALL matched players
fill_vals = merged[merged["overall"].notna()].groupby("broad_pos")["overall"].median()
print("\n=== POSITIONAL MEDIAN OVERALLS (fill values) ===")
print(fill_vals)

# Fill unmatched players
def fill_row(row):
    if pd.notna(row["overall"]):
        return row
    pos = row["broad_pos"]
    fill = fill_vals.get(pos, fill_vals.median())
    row["overall"] = fill
    row["match_type"] = "positional_fill"
    # Fill attribute columns with positional medians too
    for col in ["pace", "shooting", "passing", "dribbling", "defending", "physic"]:
        if pd.isna(row.get(col)):
            row[col] = merged[
                merged["overall"].notna() & (merged["broad_pos"] == pos)
            ][col].median()
    return row

merged = merged.apply(fill_row, axis=1)

print("\n=== FINAL COVERAGE ===")
print(merged["match_type"].value_counts())
print(f"\nAll players have overall: {merged['overall'].notna().all()}")

# Final sanity check per team
team_coverage = merged.groupby("team").apply(
    lambda g: (g["match_type"] != "positional_fill").sum()
).sort_values()

print("\n=== TEAMS WITH FEWEST REAL FC26 MATCHES (most reliant on fills) ===")
print(team_coverage.head(10))

merged.to_csv("data/processed/players_master.csv", index=False)
print("\nPhase 2 complete. Saved → data/processed/players_master.csv")