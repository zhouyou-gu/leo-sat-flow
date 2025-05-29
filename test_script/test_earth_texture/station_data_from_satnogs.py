import time, re, requests, pandas as pd
from urllib.parse import urlparse, parse_qs

BASE = "https://network.satnogs.org/api"
PAGE_SIZE = 1000  # try to fetch all stations in one call if allowed

def get_all_stations():
    # Attempt single large-page fetch first
    url = f"{BASE}/stations/?format=json&page_size={PAGE_SIZE}"
    rows = []
    session = requests.Session()

    while url:
        try:
            resp = session.get(url, timeout=(5, 1000))
            resp.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if resp.status_code == 500:
                # retry with backoff
                for attempt in range(4):
                    delay = 2 ** attempt
                    print(f"500 at {url}; retry in {delay}s…")
                    time.sleep(delay)
                    try:
                        resp = session.get(url, timeout=(5, 30))
                        resp.raise_for_status()
                        break
                    except requests.exceptions.HTTPError:
                        continue
                else:
                    print(f"Persistent 500 at {url}, skipping page")
                    # advance to next page manually
                    parsed = urlparse(url)
                    qs = parse_qs(parsed.query)
                    next_page = int(qs.get("page", ["1"])[0]) + 1
                    url = re.sub(r"page=\d+", f"page={next_page}", url)
                    continue
            else:
                raise
        data = resp.json()
        rows.extend(data if isinstance(data, list) else data.get("results", []))
        print(f"Fetched {len(rows)} records…")

        # Determine next page
        parsed = urlparse(url)
        qs = parse_qs(parsed.query)
        if "page_size" in qs:
            # if large-page succeeded, break
            break
        page = int(qs.get("page", ["1"])[0]) + 1
        url = re.sub(r"page=\d+", f"page={page}", url)

        if not data:
            url = None

    df = pd.DataFrame(rows)
    df.to_csv("satnogs_stations.csv", index=False)
    print(f"✓ Saved {len(df)} stations")
    return df

if __name__ == "__main__":
    stations_df = get_all_stations()
    print(stations_df.size)
