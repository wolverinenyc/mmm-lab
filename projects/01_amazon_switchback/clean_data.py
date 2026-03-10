import pandas as pd

RAW_PATH = '/Users/jonathan/Desktop/amazon_ads_temp/switchback_experiment_manual/Omamori Finances 3.xlsx - Ad Experiment (1).csv'
OUTPUT_PATH = 'data/amazon_switchback.csv'


def clean_switchback_data(raw_path: str) -> pd.DataFrame:
    """Load raw Amazon switchback experiment CSV and return cleaned DataFrame."""
    raw = pd.read_csv(raw_path, usecols=[0, 1, 2, 3, 4])
    raw = raw.dropna(subset=['Date'])

    for col in ['Ad Spend', 'Total Sales', 'Ad Attributed Sales']:
        raw[col] = raw[col].str.replace('$', '').str.replace(',', '').astype(float)

    raw['Date'] = pd.to_datetime(raw['Date'])
    raw['Treated'] = raw['Treated'].astype(int)

    df = raw.rename(columns={
        'Date': 'date',
        'Ad Spend': 'ad_spend',
        'Total Sales': 'total_sales',
        'Ad Attributed Sales': 'ad_attributed_sales',
        'Treated': 'treated',
    })

    df['geo'] = 'store_1'
    df['population'] = 1

    return df


if __name__ == '__main__':
    df = clean_switchback_data(RAW_PATH)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_PATH}")
