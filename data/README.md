# Data

The raw dataset is **not** tracked in version control. Download it with:

```bash
python scripts/download_data.py
```

This fetches the **Telco Customer Churn** dataset (7,043 customers, 21 columns)
and stores it at `data/raw/telco_churn.csv`.

Each row is a customer; the target column `Churn` indicates whether the customer
left during the last month. Features cover demographics, subscribed services,
contract type and billing information.

Source: IBM sample dataset, widely used as a public churn-modelling benchmark.
