import requests
from bs4 import BeautifulSoup
import pandas as pd
import os

URL = "https://en.wikipedia.org/wiki/2026_FIFA_World_Cup_squads"
OUTPUT = "data/raw/squads.csv"

def scrape_squads():
    print("Fetching Wikipedia squads page...")
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(URL, headers=headers)
    soup = BeautifulSoup(resp.text, "lxml")

    players = []
    current_team = None

    for tag in soup.find_all(["h3", "table"]):
        if tag.name == "h3":
            current_team = tag.text.strip()

        elif tag.name == "table" and current_team:
            for row in tag.find_all("tr", class_="nat-fs-player"):
                cols = row.find_all(["th", "td"])
                if len(cols) < 6:
                    continue
                try:
                    no      = cols[0].text.strip()
                    pos     = cols[1].text.strip()
                    name    = cols[2].text.strip()
                    dob     = cols[3].find("span", class_="bday")
                    dob     = dob.text.strip() if dob else cols[3].text.strip()
                    caps    = cols[4].text.strip()
                    goals   = cols[5].text.strip()
                    all_links = cols[6].find_all("a")
                    club = all_links[-1].text.strip() if all_links else cols[6].text.strip()

                    players.append({
                        "team": current_team,
                        "no": no,
                        "position": pos,
                        "name": name,
                        "dob": dob,
                        "caps": caps,
                        "goals": goals,
                        "club": club,
                    })
                except Exception:
                    continue

    df = pd.DataFrame(players)
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    # clean position
    df["position"] = df["position"].str.replace(r"^\d", "", regex=True)

    # clean name
    df["name"] = df["name"].str.replace(r"\s*\(captain\)", "", regex=True).str.strip()
    df.to_csv(OUTPUT, index=False)
    print(f"Done — {len(df)} players from {df['team'].nunique()} teams")
    print(df.head(10))
    return df

if __name__ == "__main__":
    scrape_squads()