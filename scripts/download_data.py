import urllib.request
from pathlib import Path

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
DEST = Path(__file__).resolve().parents[1] / "data" / "raw" / "telco_churn.csv"


def main():
    DEST.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset -> {DEST}")
    urllib.request.urlretrieve(URL, DEST)
    print("Done.")


if __name__ == "__main__":
    main()
