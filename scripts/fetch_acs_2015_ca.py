import os, sys, csv, json, time
import requests
from pathlib import Path

OUT = Path("data/external/census/acs2015_ca_tracts.csv")

# Accept optional API key (faster / fewer rate limits). Works without, just slower.
API_KEY = os.getenv("CENSUS_API_KEY", "")

# Variables (2015 ACS 5-year):
# Pop, Median HH Income, Median Rent, Median Home Value
# Educational attainment counts: Bachelor's, Master's, Professional, PhD
# Labor force / Unemployed to compute unemployment rate
VARS = [
  "NAME",
  "B01003_001E",  # pop total
  "B19013_001E",  # median household income
  "B25064_001E",  # median gross rent
  "B25077_001E",  # median home value
  "B15003_022E",  # bachelor's
  "B15003_023E",  # master's
  "B15003_024E",  # professional
  "B15003_025E",  # doctorate
  "B23025_003E",  # labor force
  "B23025_005E",  # unemployed
]

BASE = "https://api.census.gov/data/2015/acs/acs5"

# All 58 CA county FIPS (3-digit) — stable and avoids having to fetch list dynamically.
CA_COUNTIES = [
  "001","003","005","007","009","011","013","015","017","019","021","023","025","027","029","031",
  "033","035","037","039","041","043","045","047","049","051","053","055","057","059","061","063",
  "065","067","069","071","073","075","077","079","081","083","085","087","089","091","093","095",
  "097","099","101","103","105","107","109","111","113","115"
]

def fetch_tracts_for_county(county):
    params = {
        "get": ",".join(VARS),
        "for": "tract:*",
        "in": f"state:06 county:{county}",
    }
    if API_KEY:
        params["key"] = API_KEY
    r = requests.get(BASE, params=params, timeout=60)
    r.raise_for_status()
    return r.json()

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    header_written = False
    rows_written = 0

    with OUT.open("w", newline="") as f:
        wr = csv.writer(f)
        for i, c in enumerate(CA_COUNTIES, 1):
            # polite pacing helps when no API key
            if not API_KEY and i > 1:
                time.sleep(0.5)
            data = fetch_tracts_for_county(c)
            header, *rows = data
            # append state, county, tract columns are already in response
            if not header_written:
                wr.writerow(header)  # ["NAME",..., "state","county","tract"]
                header_written = True
            for r in rows:
                wr.writerow(r)
                rows_written += 1

    print(f"✅ wrote {OUT} with {rows_written} tract rows")

if __name__ == "__main__":
    try:
        main()
    except requests.HTTPError as e:
        print("HTTP error:", e, file=sys.stderr)
        sys.exit(1)
