import requests
import json
import pandas as pd
import os
import time

OUTPUT = "data/raw/understat_player_stats.csv"

LEAGUES = {
    "EPL":        "ENG-Premier League",
    "La_liga":    "ESP-La Liga",
    "Bundesliga": "GER-Bundesliga",
    "Serie_A":    "ITA-Serie A",
    "Ligue_1":    "FRA-Ligue 1",
    "RFPL":       "RUS-Premier League",
}

SEASON = "2025"
CUTOFF = "2026-06-10"

def fetch_league(league_key, league_label):
    headers = {
        "User-Agent": "Mozilla/5.0",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": f"https://understat.com/league/{league_key}/{SEASON}",
    }
    resp = requests.post(
        "https://understat.com/main/getPlayersStats/",
        headers=headers,
        data={"league": league_key, "season": SEASON},
    )
    data = resp.json()
    players = data.get("players", [])
    df = pd.DataFrame(players)
    df["league"] = league_label
    print(f"  {league_label}: {len(df)} players")
    return df

def main():
    if os.path.exists(OUTPUT):
        print(f"Already exists: {OUTPUT} — delete to re-scrape")
        return

    all_dfs = []
    for league_key, league_label in LEAGUES.items():
        print(f"Fetching {league_label}...")
        df = fetch_league(league_key, league_label)
        all_dfs.append(df)
        time.sleep(3)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(OUTPUT, index=False)
    print(f"\nDone — {len(combined)} rows saved to {OUTPUT}")
    print(combined.columns.tolist())
    print(combined.head(3))

if __name__ == "__main__":
    main()