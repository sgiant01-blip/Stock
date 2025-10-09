# -*- coding: utf-8 -*-
"""
KRX 종목별 OHLCV + 거래대금 + 상장주식수 + 업종(Sector) 수집 스크립트
    - 연도별 수집 + 병렬 처리 → 속도와 안정성 확보
    - 거래대금 보정: pykrx가 거래대금을 안 줄 경우 거래량 × 종가로 직접 계산
    - 상장주식수 보정: cap.empty일 경우 shares=None으로 처리
    - 누락 컬럼 보정: "trading value", "shares", "Sector"가 없을 경우 자동 생성
    - 중간 저장: 연도별 CSV/Parquet 저장 → 도중에 끊겨도 이어서 가능
    - 최종 병합: 모든 연도 데이터를 합쳐 data.KRX.csv와 data.KRX.parquet 생성
"""

import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# ─────────────────────────────────────────────
# 1. 설정
# ─────────────────────────────────────────────
data_Name   = "data.KRX"
start_year  = 2024
end_year    = 2025
last_date   = "20250927"
MAX_WORKERS = 4   # 동시에 실행할 스레드 수 (5~10 권장)

# ─────────────────────────────────────────────
# 2. 종목 코드 조회
# ─────────────────────────────────────────────
kospi_codes  = stock.get_market_ticker_list(last_date, market="KOSPI")
kosdaq_codes = stock.get_market_ticker_list(last_date, market="KOSDAQ")
codes        = kospi_codes + kosdaq_codes

# ─────────────────────────────────────────────
# 3. Sector 매핑 (엑셀 파일 불러오기)
# ─────────────────────────────────────────────
sector_df = pd.read_excel("KRX.Sector.xlsx")
sector_df["종목코드"] = sector_df["종목코드"].astype(str).str.zfill(6)
sector_map = dict(zip(sector_df["종목코드"], sector_df["업종명"]))

# ─────────────────────────────────────────────
# 4. 개별 종목 수집 함수
# ─────────────────────────────────────────────
def fetch_stock_data(code, start_date, end_date):
    try:
        name   = stock.get_market_ticker_name(code)
        market = "KOSPI" if code in kospi_codes else "KOSDAQ"

        # OHLCV
        ohlcv = stock.get_market_ohlcv_by_date(start_date, end_date, code)
        if ohlcv.empty:
            return None
        ohlcv.reset_index(inplace=True)

        # 거래대금 없으면 직접 계산
        if "거래대금" not in ohlcv.columns and "거래대금(원)" not in ohlcv.columns:
            ohlcv["거래대금"] = ohlcv["거래량"] * ohlcv["종가"]

        # 컬럼명 통일
        ohlcv.rename(columns={
            "날짜": "date",
            "시가": "open",
            "고가": "high",
            "저가": "low",
            "종가": "close",
            "거래량": "volume",
            "거래대금": "trading value",
            "거래대금(원)": "trading value"
        }, inplace=True)

        # 상장주식수
        cap = stock.get_market_cap_by_date(start_date, end_date, code)
        if not cap.empty and "상장주식수" in cap.columns:
            cap.reset_index(inplace=True)
            cap.rename(columns={"날짜": "date", "상장주식수": "shares"}, inplace=True)
            df = pd.merge(ohlcv, cap[["date", "shares"]], on="date", how="left")
        else:
            # cap 데이터가 없으면 shares=None
            df = ohlcv.copy()
            df["shares"] = None

        # change 계산
        df["change"] = (df["close"] - df["close"].shift(1)) / df["close"].shift(1).fillna(0)

        # 메타 정보
        df["Code"]    = code
        df["Company"] = name
        df["Market"]  = market
        df["Sector"]  = sector_map.get(code, None)

        # 누락 컬럼 보정
        for col in ["trading value", "shares", "Sector"]:
            if col not in df.columns:
                df[col] = None

        return df

    except Exception as e:
        return f"ERROR::{code}::{e}"

# ─────────────────────────────────────────────
# 5. 연도별 병렬 수집
# ─────────────────────────────────────────────
errors = []
all_years = []

for year in range(start_year, end_year + 1):
    print(f"\n📊 {year}년 데이터 수집 시작")
    start_date = f"{year}0101"
    end_date   = f"{year}1231"

    result_list = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(fetch_stock_data, code, start_date, end_date): code for code in codes}

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"{year} 진행", ncols=80):
            res = future.result()
            if res is None:
                continue
            elif isinstance(res, str) and res.startswith("ERROR::"):
                _, code, reason = res.split("::", 2)
                errors.append((year, code, reason))
            else:
                result_list.append(res)

    # 연도별 저장
    if result_list:
        year_df = pd.concat(result_list, ignore_index=True)

        cols = ["Code", "Company", "Market", "Sector", "date",
                "open", "high", "low", "close",
                "volume", "trading value", "shares", "change"]

        # 누락 컬럼 보정
        for c in cols:
            if c not in year_df.columns:
                year_df[c] = None

        year_df = year_df[cols]

        year_csv  = f"{data_Name}_{year}.csv"
        year_parq = f"{data_Name}_{year}.parquet"

        year_df.to_csv(year_csv, index=False, encoding="utf-8-sig")
        year_df.to_parquet(year_parq, index=False)

        print(f"📁 {year}년 CSV 저장 완료: {year_csv}")
        print(f"📁 {year}년 Parquet 저장 완료: {year_parq}")

        all_years.append(year_df)

# ─────────────────────────────────────────────
# 6. 전체 병합 및 저장
# ─────────────────────────────────────────────
if all_years:
    final_df = pd.concat(all_years, ignore_index=True)
    final_df.to_csv(f"{data_Name}.csv", index=False, encoding="utf-8-sig")
    final_df.to_parquet(f"{data_Name}.parquet", index=False)
    print(f"\n✅ 전체 CSV 저장 완료: {data_Name}.csv")
    print(f"✅ 전체 Parquet 저장 완료: {data_Name}.parquet")

# 에러 로그
if errors:
    with open(f"{data_Name}.error.log", "w", encoding="utf-8") as f:
        for year, code, reason in errors:
            f.write(f"{year}\t{code}\t{reason}\n")
    print(f"❌ 에러 로그 저장: {data_Name}.error.log ({len(errors)}건)")
else:
    print("✅ 모든 종목 정상 수집됨. 에러 없음.")