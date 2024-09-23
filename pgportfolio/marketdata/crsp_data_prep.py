import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt


def prepare_crsp_data(csv_file, feature_number=3, window_size=50, coin_number=11, start_date=None, end_date=None):
    print("Step 1: Loading CRSP data")
    df = pd.read_csv(csv_file)
    print(f"Loaded {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {df.columns.tolist()}")
    print(df.head())
    print("\n" + "=" * 50 + "\n")

    print("Step 2: Data preprocessing")
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    df = df.sort_values(['date', 'PERMNO'])
    if start_date:
        df = df[df['date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['date'] <= pd.to_datetime(end_date)]
    print(f"Data range: {df['date'].min()} to {df['date'].max()}")
    print(f"Number of unique stocks: {df['PERMNO'].nunique()}")
    print("\n" + "=" * 50 + "\n")

    print("Step 3: Feature selection and calculation")
    df = df[['date', 'PERMNO', 'PRC', 'VOL', 'SHROUT']]
    df.columns = ['date', 'permno', 'close', 'volume', 'shares_outstanding']
    df['market_cap'] = df['close'].abs() * df['shares_outstanding']
    df['high'] = df['close'] * 1.01
    df['low'] = df['close'] * 0.99
    print(df.head())
    print(f"Features created: {df.columns.tolist()}")
    print("\n" + "=" * 50 + "\n")

    print("Step 4: Selecting top stocks")
    top_stocks = df.groupby('permno')['market_cap'].mean().nlargest(coin_number).index
    df = df[df['permno'].isin(top_stocks)]
    print(f"Selected top {coin_number} stocks: {top_stocks.tolist()}")
    print("\n" + "=" * 50 + "\n")

    print("Step 5: Pivoting data")
    df_pivoted = df.pivot(index='date', columns='permno', values=['close', 'high', 'low', 'volume'])
    df_pivoted.columns = [f'{col[1]}_{col[0]}' for col in df_pivoted.columns]
    df_pivoted = df_pivoted.ffill()
    print(f"Pivoted data shape: {df_pivoted.shape}")
    print(df_pivoted.head())
    print("\n" + "=" * 50 + "\n")

    print("Step 6: Preparing 3D array")
    features = ['close', 'high', 'low', 'volume'][:feature_number]
    assets = top_stocks
    time_steps = len(df_pivoted)
    data = np.zeros((len(assets), len(features), time_steps))
    for i, asset in enumerate(assets):
        for j, feature in enumerate(features):
            data[i, j, :] = df_pivoted[f'{asset}_{feature}'].values
    print(f"3D array shape: {data.shape}")
    print("\n" + "=" * 50 + "\n")

    print("Step 7: Normalizing data")
    scaler = MinMaxScaler()
    for i in range(len(assets)):
        data[i] = scaler.fit_transform(data[i].T).T
    print("Data normalized using MinMaxScaler")
    print("\n" + "=" * 50 + "\n")

    print("Step 8: Visualization")
    plt.figure(figsize=(12, 6))
    for i in range(min(3, len(assets))):
        plt.plot(df_pivoted.index, data[i, 0, :], label=f'Stock {assets[i]}')
    plt.title('Normalized Closing Prices for Top 3 Stocks')
    plt.legend()
    plt.savefig('normalized_prices.png')
    plt.close()
    print("Saved visualization to 'normalized_prices.png'")

    return data, df_pivoted.index


if __name__ == "__main__":
    # Replace with your actual CRSP CSV file path
    csv_file = '../../OneDrive - University of Edinburgh/Development/Adaptive_market_portfolio/data/processed/crsp_market_data_v2.csv'

    data, dates = prepare_crsp_data(csv_file,
                                    feature_number=3,
                                    window_size=50,
                                    coin_number=11,
                                    start_date='2010-01-01',
                                    end_date='2020-12-31')

    print("\nFinal Output:")
    print(f"Data shape: {data.shape}")
    print(f"Date range: {dates[0]} to {dates[-1]}")