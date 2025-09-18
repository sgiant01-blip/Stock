import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shutil

# ──────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────
FileName = "data.sample.399"
dataName = FileName + ".csv"
backup_path = FileName + ".bak.csv"

# ──────────────────────────────────────────────────
def daily_to_weekly_grouped_safe(df):
    df = df.copy()
    # Date → datetime (숫자/문자열 모두 커버)
    if pd.api.types.is_numeric_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], unit='d', origin='1899-12-30', errors='coerce')
    else:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    # NaT 제거 + 정렬
    df = df.dropna(subset=['Date'])
    df = df.sort_values(['Code','Date'])
    weekly_list = []
    for (code, company), grp in df.groupby(['Code','Company']):
        grp = grp.set_index('Date')
        o = grp['Open'].resample('W-FRI').first()
        h = grp['High'].resample('W-FRI').max()
        l = grp['Low'].resample('W-FRI').min()
        c = grp['Close'].resample('W-FRI').last()
        v = grp['Volume'].resample('W-FRI').sum()
        wk = pd.DataFrame({
            'Code': code,
            'Company': company,
            'Open':   o,
            'High':   h,
            'Low':    l,
            'Close':  c,
            'Volume': v
        }).dropna(subset=['Open','High','Low','Close'])
        if not wk.empty:
            wk['MA20']   = wk['Close'].rolling(20).mean()
            wk['MA60']   = wk['Close'].rolling(60).mean()
            wk['MA150']  = wk['Close'].rolling(150).mean()
            wk['MA200']  = wk['Close'].rolling(200).mean()
            wk['Low_52w']= wk['Close'].rolling(52).min()
            wk = wk.reset_index().rename(columns={'Date':'Week'})
            weekly_list.append(wk)
    if not weekly_list:
        raise ValueError("주간 변환 결과가 없습니다.")
    return pd.concat(weekly_list, ignore_index=True)

def simple_stage_classification(weekly_df):
    out = []
    for (code, company), grp in weekly_df.groupby(['Code','Company']):
        grp = grp.sort_values('Week').copy()
        ma200 = grp['Close'].rolling(200).mean()
        stages = []
        prev_ma = ma200.shift(1)
        for close, cur_ma, pm in zip(grp['Close'], ma200, prev_ma):
            if pd.isna(cur_ma) or pd.isna(pm):
                stages.append(0)
            elif close > cur_ma and cur_ma > pm:
                stages.append(2)
            elif close > cur_ma and cur_ma <= pm:
                stages.append(3)
            elif close < cur_ma and cur_ma < pm:
                stages.append(4)
            else:
                stages.append(1)
        grp['Stage'] = stages
        out.append(grp)
    return pd.concat(out, ignore_index=True)

# 1) 원본 일간 데이터 읽기
daily = pd.read_csv(dataName, encoding="utf-8-sig")

# 2) Date → datetime, Week 추가
daily['Date'] = pd.to_datetime(daily['Date'], errors='coerce')
daily = daily.dropna(subset=['Date'])
daily = daily[~((daily['Close'] == 0) & (daily['Volume'] == 0))]
daily['Week'] = daily['Date'].dt.to_period('W-FRI')

# 3) 주간 집계 & Stage 계산
weekly  = daily_to_weekly_grouped_safe(daily)
weekly_with_stage = simple_stage_classification(weekly)
weekly_with_stage['Week'] = weekly_with_stage['Week'].dt.to_period('W-FRI')

# 4) 일간 데이터에 merge (원본 Stage 있으면 suffix로 남기고, 아래서 덮어쓰기)
merged = pd.merge(
    daily,
    weekly_with_stage[['Code','Company','Week','Stage']],
    on=['Code','Company','Week'],
    how='left',
    suffixes=('','_new')
)

# 5) Stage 컬럼 업데이트/신규 추가
#   - daily에 기존 'Stage'가 있으면 그 값은 merged['Stage'], 
#   - 주간 계산값은 merged['Stage_new']
merged['Stage'] = merged['Stage_new'].fillna(merged.get('Stage', 0)).astype(int)

# 6) 정리: 불필요 컬럼 제거, 순서 재조정
merged = merged.drop(columns=['Stage_new','Week'])

# 7) 결과 저장
shutil.copyfile(dataName, backup_path)
merged.to_csv(dataName, index=False, encoding="utf-8-sig")

print(f"✅ 원본 덮어쓰기 완료 (백업: {backup_path})")
