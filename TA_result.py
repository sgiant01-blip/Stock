import os
import pandas as pd
import pandas_ta as ta
from tqdm.auto import tqdm
from datetime import datetime

# ──────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────
FILE_IN      = 'DataKRX.csv'
PARQUET_FILE = 'TA_values.KRX.parquet'
CSV_FILE     = 'TA_values.KRX.csv'
DATE_COL     = 'Date'
lengths      = range(2, 11)  # 2~10일

required_cols = [
    'Code','Date','Open','High','Low',
    'Close','Volume','Change','Stage'
]

# ──────────────────────────────────────────────────
# 1) 과거 결과 로드
# ──────────────────────────────────────────────────
if os.path.exists(PARQUET_FILE):
    df_old    = pd.read_parquet(PARQUET_FILE)
    df_old    = df_old.loc[:, ~df_old.columns.duplicated()]
    last_date = df_old[DATE_COL].max()
else:
    df_old    = pd.DataFrame()
    last_date = datetime(1900, 1, 1)

# ──────────────────────────────────────────────────
# 2) 신규 데이터 로드 & NA·0값 필터링
# ──────────────────────────────────────────────────
df_raw = pd.read_csv(
    FILE_IN,
    parse_dates=[DATE_COL],
    dtype={'Code': str},          # ← Code를 string으로 고정
    encoding='utf-8-sig'
)

# (1) 필수 컬럼 중 NA 있으면 제거
df_raw = df_raw.dropna(subset=required_cols)

# (2) Stage 타입 변환, 날짜 정렬
df_raw['Stage'] = df_raw['Stage'].astype('category')
df_raw = df_raw.sort_values(['Code', DATE_COL]).reset_index(drop=True)

# (3) 가격/거래량 0 이하 제거
df_raw = df_raw.query("Close > 0 and Volume > 0").reset_index(drop=True)

# (4) 마지막 날짜 이후 신규만
df_new = df_raw[df_raw[DATE_COL] > last_date].reset_index(drop=True)
if df_new.empty:
    print("신규 데이터가 없습니다. 종료합니다.")
    exit()

# ──────────────────────────────────────────────────
# 3) 지표 정의
# ──────────────────────────────────────────────────
single_funcs = [
    ('EFI',    lambda df,L: ta.efi(close=df['Close'], volume=df['Volume'], length=L)),
    ('NVI',    lambda df,L: ta.nvi(close=df['Close'], volume=df['Volume'], length=L)),
    ('EMA',    lambda df,L: ta.ema(close=df['Close'], length=L)),
    ('VolEMA', lambda df,L: ta.ema(close=df['Volume'], length=L)),
    ('ROC',    lambda df,L: ta.roc(close=df['Close'], length=L)),
    ('RSI',    lambda df,L: ta.rsi(close=df['Close'], length=L)),
    ('CMF',    lambda df,L: ta.cmf(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], length=L)),
    ('TRIX',   lambda df,L: ta.trix(close=df['Close'], length=2*L, signal=L)),
    ('UO',     lambda df,L: ta.uo(high=df['High'], low=df['Low'], close=df['Close'], fast=L, medium=2*L, slow=3*L)),
    ('WillR',  lambda df,L: ta.willr(high=df['High'], low=df['Low'], close=df['Close'], length=L)),
    ('PSI',    lambda df,L: ta.psl(close=df['Close'], length=L)),
    ('Disparity', lambda df,L: df['Close'] - ta.ema(close=df['Close'], length=L)),
    ('ATR',    lambda df,L: ta.atr(high=df['High'], low=df['Low'], close=df['Close'], length=L)),
    ('CCI',    lambda df,L: ta.cci(high=df['High'], low=df['Low'], close=df['Close'], length=L)),
    ('MassIndex', lambda df,L: ta.massi(high=df['High'], low=df['Low'], fast=L, slow=int(2.5*L))),
    ('EOM',    lambda df,L: ta.eom(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], length=L)),
    ('CO',     lambda df,L: ta.adosc(high=df['High'], low=df['Low'], close=df['Close'], volume=df['Volume'], fast=3, slow=2*L)),
    ('VolumeRatio', lambda df,L: df['Volume'] / ta.sma(df['Volume'], length=L)),
]

multi_funcs = [
    ('BBANDS',     lambda df,L: ta.bbands(close=df['Close'], length=L)),
    ('AROON',      lambda df,L: ta.aroon(high=df['High'], low=df['Low'], length=L)),
    # ('DMI',        lambda df,L: ta.dm(high=df['High'], low=df['Low'], length=L)),
    ('ElderRay',   lambda df,L: ta.eri(high=df['High'], low=df['Low'], close=df['Close'], length=L)),
    ('PPO',        lambda df,L: ta.ppo(close=df['Close'], fast=1.3*L, slow=3*L, signal=L)),
    ('PVO',        lambda df,L: ta.pvo(volume=df['Volume'], fast=1.3*L, slow=2.6*L, signal=L)),
    ('sRSI',       lambda df,L: ta.stochrsi(close=df['Close'], length=L, rsi_length=L, k=3, d=3)),
    ('MACD',       lambda df,L: ta.macd(close=df['Close'], fast=1.3*L, slow=3*L, signal=L)),
    ('KC',         lambda df,L: ta.kc(high=df['High'], low=df['Low'], close=df['Close'], length=L)),
    ('ADX',        lambda df,L: ta.adx(high=df['High'], low=df['Low'], close=df['Close'], length=L)),
]

extra_funcs = [
    ('PSAR',     lambda df: ta.psar(high=df['High'], low=df['Low'])),
    ('ADLine',   lambda df: pd.DataFrame({'ADLine': ta.ad(df['High'], df['Low'], df['Close'], df['Volume'])})),
    ('OBV',      lambda df: pd.DataFrame({'OBV': ta.obv(close=df['Close'], volume=df['Volume'])})),
    ('PVT',      lambda df: pd.DataFrame({'PVT': ta.pvt(close=df['Close'], volume=df['Volume'])})),
    ('Donchian', lambda df: ta.donchian(high=df['High'], low=df['Low'])),
    ('Pivot',    lambda df: pd.DataFrame({
                      'Pivot': (df['High'].shift(1)+df['Low'].shift(1)+df['Close'].shift(1))/3,
                      'Pivot_R1':    2*((df['High'].shift(1)+df['Low'].shift(1)+df['Close'].shift(1))/3) - df['Low'].shift(1),
                      'Pivot_S1':    2*((df['High'].shift(1)+df['Low'].shift(1)+df['Close'].shift(1))/3) - df['High'].shift(1),
                      'Pivot_R2':    ((df['High'].shift(1)+df['Low'].shift(1)+df['Close'].shift(1))/3)
                               + (df['High'].shift(1)-df['Low'].shift(1)),
                      'Pivot_S2':    ((df['High'].shift(1)+df['Low'].shift(1)+df['Close'].shift(1))/3)
                               - (df['High'].shift(1)-df['Low'].shift(1)),
    })),
]

# ──────────────────────────────────────────────────
# 4) 전체 작업량 계산
# ──────────────────────────────────────────────────
codes      = df_new['Code'].unique()
n_codes    = len(codes)
n_single   = len(single_funcs) * len(lengths)
n_multi    = len(multi_funcs)  * len(lengths)
n_extras   = len(extra_funcs)
total_tasks = n_codes * (n_single + n_multi + n_extras)

# ──────────────────────────────────────────────────
# 5) 계산 루프 & 단편화 방지
# ──────────────────────────────────────────────────
buffers = []
pbar    = tqdm(total=total_tasks, desc='전체 보조지표 계산', unit='step')

for code in codes:
    df_grp = df_new[df_new['Code']==code].sort_values(DATE_COL).reset_index(drop=True)
    if df_grp.empty:
        tqdm.write(f"[SKIP] {code} – 유효 데이터 없음")
        continue

    pieces = [df_grp]

    # single funcs
    for name, fn in single_funcs:
        for L in lengths:
            try:
                s = fn(df_grp, L)
                if s is not None:
                    pieces.append(s.rename(f'{name}_{L}'))
            except Exception:
                pass
            pbar.update()

    # multi funcs
    for name, fn in multi_funcs:
        for L in lengths:
            try:
                dfc = fn(df_grp, L)
                if dfc is not None:
                    pieces.append(dfc)
            except Exception:
                pass
            pbar.update()

    # extras
    for name, fn in extra_funcs:
        try:
            dfc = fn(df_grp)
            if dfc is not None:
                pieces.append(dfc)
        except Exception:
            pass
        pbar.update()

    df_grp = pd.concat(pieces, axis=1)
    buffers.append(df_grp)

pbar.close()

# ──────────────────────────────────────────────────
# 6) 결과 병합 & 저장
# ──────────────────────────────────────────────────
df_final = pd.concat(buffers, ignore_index=True)
df_final.to_parquet(PARQUET_FILE, index=False)
# df_final.to_csv(CSV_FILE, index=False, encoding='cp949')

print(f"완료: 총 {len(df_final)}개 행 저장됨.")