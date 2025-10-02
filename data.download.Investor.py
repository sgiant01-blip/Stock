import os
import pandas as pd
from pykrx import stock
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

"""
PyKRX 기반 투자자별 매매 동향 수집 (속도 최적화 + 컬럼 정규화 + 깔끔한 진행바)
- IN_PARQ/OUT_FILE 상수 최상단 정의
- 종목별/날짜별 외국인, 기관, 법인, 개인 매매금액 수집
- 특수종목(스팩·리츠·우선주·ETN) 분리
- 일반종목만 멀티스레드로 수집
- 원본 IN_PARQ에 외국인/기관/법인/개인 컬럼 병합 후 저장
- '외국인' vs '외국인 합계' 등 표기 차이를 모두 표준 컬럼으로 매핑
- 누락 컬럼은 0으로 채움
- 멀티스레드로 속도 향상
- 진행 중 불필요한 콘솔 메시지 제거 (에러는 파일로 저장)
- 에러 로그에 코드별 ‘사유(reason)’ 출력
"""

# ──────────────────────────────────────────────────────────────────────────────
# 0. 경로/출력 상수
# ──────────────────────────────────────────────────────────────────────────────
data_Name   = "data.KRX"
IN_PARQ   = data_Name + ".parquet"
IN_CSV   = data_Name + ".csv"
OUT_PARQ    = data_Name + ".Investor.parquet"
OUT_CSV     = data_Name + ".Investor.csv"
ERROR_LOG   = data_Name + ".Investor.errors.log"

TQDM_BAR_FORMAT = "{l_bar}{bar} | {n_fmt}/{total_fmt} | {elapsed}<{remaining} | {rate_fmt}"

TARGET_COLS = ["외국인", "기관", "법인", "개인"]
COLUMN_MAP = {
    "외국인": "외국인", "외국인합계": "외국인", "외국인 합계": "외국인",
    "기관": "기관",   "기관합계": "기관",   "기관 합계": "기관",
    "법인": "법인",   "법인합계": "법인",   "법인 합계": "법인",
    "기타법인": "법인", "기타 법인": "법인",
    "개인": "개인",   "개인합계": "개인",   "개인 합계": "개인",
}

# ──────────────────────────────────────────────────────────────────────────────
# 1. 단일 종목 수집 (컬럼 정규화)
# ──────────────────────────────────────────────────────────────────────────────
def fetch_investor(code: str, start_date: str, end_date: str) -> pd.DataFrame:
    try:
        df = stock.get_market_trading_value_by_date(start_date, end_date, code)
        df = df.rename_axis("date").reset_index()
        df["Code"] = code

        src_cols = [c for c in COLUMN_MAP if c in df.columns]
        out     = df[["date", "Code"] + src_cols].rename(columns=COLUMN_MAP)

        # 누락된 컬럼은 0으로 채우기
        for col in TARGET_COLS:
            if col not in out.columns:
                out[col] = 0

        return out[["date", "Code"] + TARGET_COLS]

    except Exception:
        return pd.DataFrame()

# ──────────────────────────────────────────────────────────────────────────────
# 2. 종목 유형 분류 함수
# ──────────────────────────────────────────────────────────────────────────────
def classify_type(name: str) -> str:
    name = name.upper()
    if "스팩" in name:
        return "스팩"
    if "REIT" in name or "리츠" in name:
        return "리츠"
    if name.endswith("우"):
        return "우선주"
    if "ETN" in name:
        return "ETN"
    return "일반"

# ──────────────────────────────────────────────────────────────────────────────
# 3. 전체 수집 및 병합
# ──────────────────────────────────────────────────────────────────────────────
def collect_investor(max_workers: int = 8):
    if os.path.exists(IN_PARQ):
        krx = pd.read_parquet(IN_PARQ)
    elif os.path.exists(IN_CSV):
        krx = pd.read_csv(
            IN_CSV,
            parse_dates=['date'],
            dtype={'Code': str},
            low_memory=False
        )
    else:
        raise FileNotFoundError(f"'{IN_PARQ}' 또는 '{IN_CSV}' 파일이 없습니다.")

    # 타입 정리
    krx["date"]   = pd.to_datetime(krx["date"])
    krx["Code"]   = krx["Code"].astype(str)
    krx["Name"]   = krx["Name"].astype(str)
    krx["Market"] = krx["Market"].astype(str)

    # 종목명 기준으로 유형 분류
    code_name_df = krx[["Code", "Name"]].drop_duplicates()
    type_map     = {
        row.Code: classify_type(row.Name)
        for row in code_name_df.itertuples()
    }

    special_codes = [c for c, t in type_map.items() if t != "일반"]
    normal_codes  = [c for c, t in type_map.items() if t == "일반"]

    # 수집 기간
    start_date = krx["date"].min().strftime("%Y%m%d")
    end_date   = krx["date"].max().strftime("%Y%m%d")

    results = []
    errors  = []

    # 3-2. 특수종목은 에러 로그에만 기록
    for code in special_codes:
        errors.append((code, "특수종목"))

    # 3-3. 일반종목 멀티스레드 수집
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(fetch_investor, code, start_date, end_date): code
            for code in normal_codes
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="투자자별 매매 동향 수집",
            unit="종목",
            dynamic_ncols=True,
            smoothing=0.1,
            bar_format=TQDM_BAR_FORMAT
        ):
            code = futures[future]
            try:
                df = future.result()
                if df.empty:
                    errors.append((code, "데이터없음"))
                else:
                    results.append(df)
            except Exception as e:
                errors.append((code, str(e)))

    # 3-4. 수집 결과 병합 및 저장
    if results:
        inv_df = pd.concat(results, ignore_index=True)
        # 원본 krx와 일자·종목 코드 기준으로 병합(left join)
        merged = pd.merge(
            krx,
            inv_df,
            on=["date", "Code"],
            how="left"
        )
        merged.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        merged.to_parquet(OUT_PARQ, index=False)
        print(f"✅ {OUT_CSV} 저장 완료 (원본 + 투자자매매 컬럼 병합)")

    # 3-5. 에러 로그 기록 (코드 \t 사유)
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        for code, reason in errors:
            f.write(f"{code}\t{reason}\n")
    print(f"❌ 에러 로그: {ERROR_LOG} ({len(errors)}건 기록됨)")

# ──────────────────────────────────────────────────────────────────────────────
# 4. 엔트리 포인트
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    collect_investor(max_workers=8)
