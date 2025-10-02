# -*- coding: utf-8 -*-
"""
KRX 종목별 OHLCV + 투자자별 change(open–close) 수집 스크립트
- change = 시가(open) - 종가(close)
- 결측 change는 0으로 채움
- 진행 상황은 tqdm 프로그레스바로 표시
- 결과를 CSV와 Parquet으로 저장
- 수집 실패 종목은 error.log에 코드와 사유 기록
"""

import time
import pandas as pd
from pykrx import stock
from tqdm import tqdm

# ──────────────────────────────────────────────────────────────────────────────
# 1. 설정
# ──────────────────────────────────────────────────────────────────────────────
data_Name  = "data.KRX"
OUT_CSV    = data_Name + ".csv"
OUT_PARQ   = data_Name + ".parquet"
ERROR_LOG  = data_Name + ".error.log"

# 조회 기간 (YYYYMMDD)
start_date = "20150102"
end_date   = "20250927"

# ──────────────────────────────────────────────────────────────────────────────
# 2. 종목 코드 및 메타 정보 조회
# ──────────────────────────────────────────────────────────────────────────────
kospi_codes  = stock.get_market_ticker_list(end_date, market="KOSPI")
kosdaq_codes = stock.get_market_ticker_list(end_date, market="KOSDAQ")
codes        = kospi_codes + kosdaq_codes

def get_meta(code: str):
    """종목 코드 → (Name, Market) 반환"""
    name   = stock.get_market_ticker_name(code)
    market = "KOSPI" if code in kospi_codes else "KOSDAQ"
    return name, market

# ──────────────────────────────────────────────────────────────────────────────
# 3. 수집 루프
# ──────────────────────────────────────────────────────────────────────────────
result_list = []
errors      = []

print("📊 종목별 데이터 수집 시작:")

for code in tqdm(codes,
                 desc="수집 진행",
                 total=len(codes),
                 unit="종목",
                 ncols=80):
    try:
        name, market = get_meta(code)

        # 3-1) OHLCV 데이터
        df = stock.get_market_ohlcv(start_date, end_date, code)
        df.reset_index(inplace=True)  # '날짜' 컬럼을 일반 컬럼으로

        # 3-2) 컬럼명 통일
        df.rename(columns={
            "날짜":   "date",
            "시가":    "open",
            "고가":    "high",
            "저가":    "low",
            "종가":    "close",
            "거래량":   "volume"
        }, inplace=True)

        # 3-3) change 계산: open – close, 결측값 0
        df["change"] = (df["open"] - df["close"]).fillna(0)

        # 3-4) 메타 컬럼 추가
        df["Code"]   = code
        df["Company"]   = name
        df["Market"] = market

        result_list.append(df)

        # Optional: 서버 부담 완화
        time.sleep(0.05)

    except Exception as e:
        # 실패 종목 기록
        errors.append((code, str(e)))

# ──────────────────────────────────────────────────────────────────────────────
# 4. 결과 합치기 및 저장
# ──────────────────────────────────────────────────────────────────────────────
print("\n✅ 수집 완료. 데이터 병합 및 저장 중…")

if not result_list:
    raise RuntimeError("수집된 데이터가 하나도 없습니다.")

# 4-1) 하나의 DataFrame으로 병합
final_df = pd.concat(result_list, ignore_index=True)

# 4-2) 컬럼 순서 재정렬
cols = ["Code", "Company", "Market", "date",
        "open", "high", "low", "close",
        "volume", "change"]
final_df = final_df[cols]

# 4-3) CSV 저장
final_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
print(f"📁 CSV 저장 완료: {OUT_CSV}")

# 4-4) Parquet 저장 (pyarrow 또는 fastparquet 필요)
final_df.to_parquet(OUT_PARQ, index=False)
print(f"📁 Parquet 저장 완료: {OUT_PARQ}")

# ──────────────────────────────────────────────────────────────────────────────
# 5. 에러 로그 기록
# ──────────────────────────────────────────────────────────────────────────────
if errors:
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        for code, reason in errors:
            f.write(f"{code}\t{reason}\n")
    print(f"❌ 에러 로그 저장: {ERROR_LOG} ({len(errors)}건)")
else:
    print("✅ 모든 종목 정상 수집됨. 에러 없음.")

