# -*- coding: utf-8 -*-
"""
종목 Stage 분류 스크립트

목적 : 과거 KRX 종목별 실적(data.KRX.csv)을 기반으로
      보조지표(이동평균, RSI 등)와 수급(외국인·기관·개인·법인 매매),
      시장환경(강세/약세/중립)을 결합해 단기 매매용 Stage(돌파·추세·과열·조정·침체)를 분류

주요 단계
    1) 데이터 로딩 및 전처리
       - 결측치 처리, 타입 변환, 정렬
    2) 종목별 보조지표 계산
       - 이동평균(MA5/20/60), 거래량평균(VOL20), HHV/LLV20, 수익률, RSI14
       - share 기반 회전율(turnover), 20일 누적 수급비율
    3) 수급 
        - 외국인, 기관, 법인, 개인 매매대금 비율
        - 순매도, 순매수, 중립
    4) 시장환경 산출 (Market 단위)
       - 폭넓은 강세·약세 지표(breadth, eq_ret, vol5, net_flow)
       - 5일 누적 수익률·순매수 흐름 판단
       - 중립, 약세, 강세
    5) Stage 분류
       - 종목 흐름·수급·시장환경을 조합해 5개 국면으로 라벨링
         조정, 과열, 돌파, 확산, 침체
       - 기타 조건은 “기타” 카테고리로 분류

코드 해석
  1) net_flow 
      - net_flow > 0: 기관·외국인 등 “큰손” 자금이 순유입 중
	    → 시장 자금 분위기가 우호적, 상승 탄력 기대
      - net_flow < 0: 개인 매매 주도로 순유출 발생
	    → 시장에 자금이 빠져나가는 중, 약세 경향
   2) Breadth 
      - Breadth > 0.6
        시세 상승에 폭넓은 종목 동반 강세
        지수 상승이 견고할 가능성 높음
      - Breadth 0.4 ~ 0.6
        시장 전반적으로 방향성 불분명
        지수 레벨 신호만으로 매매하기엔 리스크
      - Breadth < 0.4
        소수 종목만 상승 주도 → 좁은 장세
        지수 회복 시도라도 대체로 불안정할 수 있음
   3) buy_pressure20
      - buy_pressure20 > 0
        최근 20일간 비개인이 개인보다 순매수한 규모가 더 크다는 뜻
        기관·외국인 주도로 주가 상승이 뒷받침될 가능성 ↑
      - buy_pressure20 < 0
        개인이 상대적으로 더 많이 순매수했다는 뜻
        비개인의 순매도 우위 → 약세 혹은 불확실 구간
      - 값의 크기로도 세기를 구분할 수 있습니다. 예를 들어
        상위 25% percentile 이상 → 강한 매수 모멘텀
        중간 구간(−25% ~ +25%) → 중립, 방향성 불명확
        하위 25% 이하 → 강한 매도 압력
주의 사항
    - pandas FutureWarning: DataFrameGroupBy.apply operated on grouping columns 무시
"""
import warnings
warnings.filterwarnings(
    "ignore",
    message="DataFrameGroupBy.apply operated on the grouping columns"
)

import os
import sys
import argparse

import pandas as pd
import numpy as np
from tqdm.auto import tqdm

tqdm.pandas()

# ──────────────────────────────────────────────────────────────────────────────
# 0) 파일 경로 및 스모크 테스트용 코드
# ──────────────────────────────────────────────────────────────────────────────
DATA_NAME     = 'data.KRX.Investor'
IN_PARQ       = f'{DATA_NAME}.parquet'
IN_CSV        = f'{DATA_NAME}.csv'
OUT_STAGE_PQ  = f'{DATA_NAME}.Stage.parquet'
OUT_STAGE_CSV = f'{DATA_NAME}.Stage.csv'
OUT_ADD_PQ    = f'{DATA_NAME}.Additional.parquet'
OUT_ADD_CSV   = f'{DATA_NAME}.Additional.csv'

SAMPLE_CODES = ['005930','000660','035420']


# ──────────────────────────────────────────────────────────────────────────────
# 1) 데이터 로딩 & 기본 전처리
# ──────────────────────────────────────────────────────────────────────────────
def load_data(parquet_path, csv_path):
    if os.path.exists(parquet_path):
        df = pd.read_parquet(parquet_path)
    elif os.path.exists(csv_path):
        df = pd.read_csv(csv_path, parse_dates=['date'], encoding='utf-8-sig')
    else:
        print(f"[Error] 입력 파일이 없습니다: {parquet_path} 또는 {csv_path}")
        sys.exit(1)

    df.columns = df.columns.str.strip()
    num_cols = ['open','high','low','close','volume','change',
                '외국인','기관','법인','개인','shares']
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0)

    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['Market','Code','date']).reset_index(drop=True)

    for c in ('Code','Market','date','close'):
        assert c in df.columns, f"필수 컬럼 누락: {c}"
    return df


# ──────────────────────────────────────────────────────────────────────────────
# 2) 종목별 피처 계산
# ──────────────────────────────────────────────────────────────────────────────
def compute_group_features(g):
    g = g.copy()
    g['ret'] = g['close'].pct_change()

    g['MA5']   = g['close'].rolling(5).mean()
    g['MA20']  = g['close'].rolling(20).mean()
    g['MA60']  = g['close'].rolling(60).mean()
    g['VOL20'] = g['volume'].rolling(20).mean()
    g['HHV20'] = g['high'].rolling(20).max()

    delta    = g['close'].diff()
    gain     = delta.clip(lower=0)
    loss     = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean().replace(0, np.nan)
    g['RSI14'] = 100 - (100 / (1 + avg_gain.div(avg_loss)))

    if 'shares' in g and g['shares'].gt(0).any():
        g['turnover']   = g['volume'] / g['shares']
        g['turnover20'] = g['turnover'].rolling(20).mean()
    else:
        g['turnover']   = pd.Series(np.nan, index=g.index)
        g['turnover20'] = pd.Series(np.nan, index=g.index)

    for col in ('외국인','기관','법인','개인'):
        if 'shares' in g and g['shares'].gt(0).any():
            ratio = g[col] / g['shares']
        else:
            ratio = pd.Series(np.nan, index=g.index)
        g[f'{col}_ratio']      = ratio
        g[f'{col}_ratio_cum20'] = ratio.rolling(20).sum()

    non_indiv = g[['외국인','기관','법인']].sum(axis=1)
    g['buy_pressure']   = non_indiv - g['개인']
    g['buy_pressure20'] = g['buy_pressure'].rolling(20).mean()

    for feat in ('MA5','RSI14','buy_pressure20'):
        assert feat in g.columns, f"{feat} 계산 누락"
    return g


# ──────────────────────────────────────────────────────────────────────────────
# 3) 시장환경 계산
# ──────────────────────────────────────────────────────────────────────────────
def compute_market_env(mkt):
    dgrp = mkt.groupby('date')

    breadth = dgrp.apply(lambda x: (x['close'] > x['MA20']).mean())
    eq_ret  = dgrp['ret'].mean()
    net_flow = dgrp.apply(
        lambda x: (x[['외국인','기관','법인']].sum(axis=1) - x['개인']).sum()
    )
    vol5 = eq_ret.rolling(5).std()

    env = pd.DataFrame({
        'date'     : breadth.index,
        'breadth'  : breadth.values,
        'eq_ret'   : eq_ret.values,
        'vol5'     : vol5.values,
        'net_flow' : net_flow.values
    })
    env['eq_ret5'] = env['eq_ret'].rolling(5).sum()

    bull = (env['breadth'] > 0.6) & (env['eq_ret5'] > 0) & (env['net_flow'] > 0)
    bear = (env['breadth'] < 0.4) & (env['eq_ret5'] < 0) & (env['net_flow'] < 0)
    env['market_env'] = np.where(bull, '강세', np.where(bear, '약세', '중립'))

    assert 'market_env' in env.columns
    env['Market'] = mkt['Market'].iloc[0]
    return env


# ──────────────────────────────────────────────────────────────────────────────
# 4) 메인 로직
# ──────────────────────────────────────────────────────────────────────────────
def main(smoke: bool = False):
    df = load_data(IN_PARQ, IN_CSV)
    orig_cols = df.columns.tolist()

    if smoke:
        df = df[df['Code'].isin(SAMPLE_CODES)]
        print(f"[SMOKE TEST] codes={SAMPLE_CODES}, rows={len(df)}")

    df = df.groupby(
        ['Market','Code'],
        group_keys=False
    ).progress_apply(compute_group_features)

    env_df = df.groupby(
        'Market',
        group_keys=False
    ).progress_apply(compute_market_env)

    df = df.merge(
        env_df[['Market','date','market_env']],
        on=['Market','date'], how='left'
    )

    df['stage'] = np.select(
        [
            df['close'] >= df['HHV20'],
            (df['MA5'] > df['MA20']) & (df['MA20'] > df['MA60']),
            df['RSI14'] > 70,
            df['close'] < df['MA20']
        ],
        ['돌파','확산','과열','조정'],
        default='침체'
    )
    df['supply'] = np.select(
        [df['buy_pressure20'] > 0, df['buy_pressure20'] < 0],
        ['순매수','순매도'],
        default='중립'
    )

    df['prev_close'] = df.groupby('Code')['close'].shift(1)
    df['adv'] = (df['close'] > df['prev_close']).astype(int)
    df['dec'] = (df['close'] < df['prev_close']).astype(int)
    adr = df.groupby('date').agg(
        adv_sum=('adv','sum'),
        dec_sum=('dec','sum'),
        total=('Code','size')
    ).reset_index()
    adr['A/D ratio'] = (adr['adv_sum'] - adr['dec_sum']) / adr['total']
    df = df.merge(adr[['date','A/D ratio']], on='date', how='left')

    stage_cols = orig_cols + ['market_env','stage','supply','A/D ratio']
    df_stage = df.loc[:, stage_cols]
    df_stage.to_parquet(OUT_STAGE_PQ, index=False)
    df_stage.to_csv(OUT_STAGE_CSV, index=False, encoding='utf-8-sig')

    add_cols = ['Code','Name','Market','date'] + [
        c for c in df.columns if c not in ('Code','Name','Market','date')
    ]
    df_add = df.loc[:, add_cols]
    df_add.to_parquet(OUT_ADD_PQ, index=False)
    df_add.to_csv(OUT_ADD_CSV, index=False, encoding='utf-8-sig')

    # ──────────────────────────────────────────────────────────────────────────
    # 최종 요약 및 A/D ratio 통계
    # ──────────────────────────────────────────────────────────────────────────
    print(f"\n▶ Stage DF: rows={df_stage.shape[0]}, cols={df_stage.shape[1]}")
    print("▶ 숫자형 피처 요약:")
    print(df_stage.select_dtypes(include='number').describe().T)

    print("\n▶ Counts by Stage:")
    print(df['stage'].value_counts().to_string())
    print("\n▶ Counts by Supply:")
    print(df['supply'].value_counts().to_string())
    print("\n▶ Counts by Market Env:")
    print(df['market_env'].value_counts().to_string())
    print("\n▶ A/D ratio 통계 요약:")
    print(df['A/D ratio'].describe().to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="대표 종목 스모크 테스트 모드"
    )
    args = parser.parse_args()
    main(smoke=args.smoke)