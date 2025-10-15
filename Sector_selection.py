import pandas as pd
import numpy as np
from scipy.stats.mstats import winsorize
from tqdm import tqdm

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
    df['vol5']   = df.groupby('Code')['ret1'].rolling(5).std().reset_index(0, drop=True)
    df['vol10']  = df.groupby('Code')['ret1'].rolling(10).std().reset_index(0, drop=True)
    df['vol_ma5']   = df.groupby('Code')['volume'].rolling(5).mean().reset_index(0, drop=True)
    df['turnover5'] = df.groupby('Code')['trading value'].rolling(5).mean().reset_index(0, drop=True)
    
    # NaN 값을 forward fill 후 0으로 채우기
    df['ret1'] = df.groupby('Code')['ret1'].ffill().fillna(0)
    df['ret20'] = df.groupby('Code')['ret20'].ffill().fillna(0)
    df['mom1m'] = df.groupby('Code')['mom1m'].ffill().fillna(0)
    df['vol5'] = df.groupby('Code')['vol5'].ffill().fillna(df['vol5'].median())
    df['vol10'] = df.groupby('Code')['vol10'].ffill().fillna(df['vol10'].median())
    df['vol_ma5'] = df.groupby('Code')['vol_ma5'].ffill().fillna(0)
    df['turnover5'] = df.groupby('Code')['turnover5'].ffill().fillna(0)
    
    return df

def winsorize_features(df: pd.DataFrame, cols: list, limits=(0.03, 0.03)) -> pd.DataFrame:
    for col in cols:
        df[col] = df.groupby('date')[col].transform(lambda x: winsorize(x, limits=limits))
    return df

def standardize_and_score(df: pd.DataFrame, score_cols: list) -> pd.DataFrame:
    for col in score_cols:
        zcol = f"{col}_z"
        df[zcol] = df.groupby('date')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)  # 0으로 나누기 방지
        )
        # NaN 값을 0으로 채우기
        df[zcol] = df[zcol].fillna(0)
    
    zcols = [f"{c}_z" for c in score_cols]
    df['score_raw'] = df[zcols].mean(axis=1)
    df['score_raw'] = df['score_raw'].fillna(0)  # 추가 안전장치
    
    return df

def apply_inverse_vol_weight(df: pd.DataFrame, vol_col='vol5', vol_floor=0.01, vol_cap=0.10) -> pd.DataFrame:
    df['vol_adj'] = df[vol_col].clip(lower=vol_floor, upper=vol_cap)
    df['inv_vol_wt'] = 1 / df['vol_adj']
    df['score'] = df['score_raw'] * df['inv_vol_wt']
    return df
