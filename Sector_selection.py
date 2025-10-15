import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from tqdm import tqdm  # tqdm import

def load_data(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Code', 'date']).reset_index(drop=True)
    return df

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    df['market_cap'] = df['close'] * df['shares']
    df = df[(df['close'] > 0) & (df['volume'] > 0)].copy()
    df['month_end'] = df['date'].dt.to_period('M').dt.to_timestamp('M')
    return df

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    df['ret1']   = df.groupby('Code')['close'].pct_change(1)
    df['ret20']  = df.groupby('Code')['close'].pct_change(20)
    df['mom1m']  = df['ret20']
    df['vol5']   = df.groupby('Code')['ret1'] .rolling(5).std().reset_index(0, drop=True)
    df['vol10']  = df.groupby('Code')['ret1'].rolling(10).std().reset_index(0, drop=True)
    df['vol_ma5']   = df.groupby('Code')['volume']       .rolling(5).mean().reset_index(0, drop=True)
    df['turnover5'] = df.groupby('Code')['trading value'].rolling(5).mean().reset_index(0, drop=True)
    return df

def winsorize_features(df: pd.DataFrame, cols: list, limits=(0.03, 0.03)) -> pd.DataFrame:
    for col in cols:
        df[col] = df.groupby('date')[col]\
                   .transform(lambda x: winsorize(x, limits=limits))
    return df

def standardize_and_score(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    for col in score_cols:
        zcol = f"{col}_z"
        df[zcol] = df.groupby('date')[col]\
                      .transform(lambda x: (x - x.mean()) / x.std())
    zcols = [f"{c}_z" for c in score_cols]
    df['score_raw'] = df[zcols].mean(axis=1)
    return df

def apply_inverse_vol_weight(df: pd.DataFrame, vol_col='vol5', 
                             vol_floor=0.01, vol_cap=0.10) -> pd.DataFrame:
    df['vol_adj'] = df[vol_col].clip(lower=vol_floor, upper=vol_cap)
    df['inv_vol_wt'] = 1 / df['vol_adj']
    df['score'] = df['score_raw'] * df['inv_vol_wt']
    return df

def backtest_strategy(df: pd.DataFrame,
                      top_n: int = 10,
                      min_volume: float = 1e4,
                      min_turnover: float = 1e6) -> pd.DataFrame:
    results = []
    # 고유 리밸런스 날짜
    rebalance_dates = sorted(df['month_end'].unique())
    # tqdm으로 진행 바 표시
    for dt in tqdm(rebalance_dates, desc="리밸런싱 진행", unit="달"):
        uni = df[df['month_end'] == dt].copy()

        uni = uni[(uni['volume'] > min_volume) &
                  (uni['trading value'] > min_turnover)]

        alive = (df[(df['Code'].isin(uni['Code'])) & (df['date'] > dt)]
                 .groupby('Code')['date'].min() > dt)
        uni = uni[uni['Code'].isin(alive[alive].index)]

        selected = uni.groupby('Sector')\
                      .apply(lambda x: x.nlargest(top_n, 'score'))\
                      .reset_index(drop=True)

        future = df[['Code','date','close']].copy()
        future['date'] = future['date'] - pd.offsets.MonthEnd(1)
        merged = pd.merge(selected, future,
                          on=['Code','date'], how='left',
                          suffixes=('','_next'))
        merged['ret_next1m'] = merged['close_next'] / merged['close'] - 1

        port_ret = merged['ret_next1m'].mean()
        results.append({'date': dt, 'ret': port_ret})

    res = pd.DataFrame(results).set_index('date')
    res['cum_ret'] = (1 + res['ret']).cumprod() - 1
    return res

def main():
    df = load_data('data.Analysis.parquet')
    df = preprocess_data(df)
    df = feature_engineering(df)

    to_wins = ['market_cap', 'turnover5', 'vol_ma5', 'mom1m']
    df = winsorize_features(df, to_wins)

    df = standardize_and_score(df, to_wins)
    df = apply_inverse_vol_weight(df)

    backtest_df = backtest_strategy(df)
    print(backtest_df)
    backtest_df.to_csv('backtest_results.csv')

if __name__ == '__main__':
    main()