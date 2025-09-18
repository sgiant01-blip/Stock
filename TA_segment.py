import pandas as pd
import numpy as np
from tqdm import tqdm
import pyarrow.parquet as pq
from dateutil.relativedelta import relativedelta

RAW_PARQUET = 'TA_values.399.parquet'
OUT_CSV     = 'TA_segmented.399.csv'

def compute_future_returns(df, days=[1,2,3]):
    # 1) 개별 일수별 수익률 계산
    for d in tqdm(days,
                  desc="▶ Compute ret_d",
                  unit="day",
                  ncols=80):
        df[f"ret_{d}"] = (
            df.groupby("Code")["Close"]
              .shift(-d)
              .div(df["Close"])
              .sub(1)
        )

    # 2) 누적 평균 수익률 계산
    for d in tqdm(days,
                  desc="▶ Compute ret_avg_d",
                  unit="day",
                  ncols=80):
        cols = [f"ret_{i}" for i in range(1, d+1)]
        df[f"ret_avg_{d}"] = df[cols].mean(axis=1)

    return df

def segment_and_summarize(df,
                          days=[1,2,3,4,5],
                          lower_pct=0.10,
                          upper_pct=0.90,
                          n_segments=50):
    base_cols = ["Code","Company","Date","Open","High","Low","Close","Volume","Change","Stage"]
    ret_cols  = [f"ret_{i}"     for i in days] + \
                [f"ret_avg_{i}" for i in days]
    indicator_cols = [c for c in df.columns if c not in base_cols + ret_cols]

    results = []
    
    # 3) 지표별 메인 루프
    for ind in tqdm(indicator_cols,
                    desc="▶ Indicators",
                    unit="indicator",
                    ncols=80):
        df_ind = df.copy()
        df_ind["rank_pct"] = df_ind[ind].rank(method="first", pct=True)
        df_mid = df_ind[df_ind["rank_pct"].between(lower_pct, upper_pct)].copy()

        try:
            df_mid["segment"] = pd.qcut(df_mid[ind],
                                        q=n_segments,
                                        labels=False,
                                        duplicates="drop")
        except ValueError:
            continue

        seg_bounds = (
            df_mid.groupby("segment")[ind]
                  .agg(["min","max"])
                  .rename(columns={"min":"segment_lo","max":"segment_hi"})
        )
        df_mid = df_mid.merge(seg_bounds, on="segment", how="left")
        df_mid[f"{ind}_dir"] = df_mid[ind].diff() > 0

        # 4) 일별 루프에도 진행바
        for d in tqdm(days,
                      desc=f"   ◼ {ind} days",
                      unit="day",
                      ncols=80,
                      leave=False):
            ret_col = f"ret_avg_{d}"
            grp = df_mid.dropna(subset=["segment", ret_col]).groupby(
                ["Stage","segment","segment_lo","segment_hi"]
            )
            agg = grp.agg(
                total_count = ("segment","size"),
                pos_match   =(f"{ind}_dir","sum"),
                mean_return =(ret_col,"mean"),
                neg3_count  =(ret_col, lambda x: (x <= -0.03).sum()),
                min_return  =(ret_col,"min")
            ).reset_index()

            agg["pos_ratio"]  = agg["pos_match"]  / agg["total_count"]
            agg["neg3_ratio"] = agg["neg3_count"] / agg["total_count"]
            agg["indicator"]  = ind
            agg["day"]        = d

            results.append(agg)

    # 5) 결과가 없으면 빈 DF 리턴
    ordered_cols = [
        "Stage","day","indicator","segment_lo","segment_hi",
        "total_count","pos_match","pos_ratio",
        "mean_return","neg3_count","neg3_ratio","min_return"
    ]
    if not results:
        return pd.DataFrame(columns=ordered_cols)

    return pd.concat(results, ignore_index=True)

def main():
    print("▶ Loading data (multithreaded)")
    table = pq.read_table(RAW_PARQUET, use_threads=True)   
    df = table.to_pandas()

    # Date 변환 & 데이터에 있는 마지막 날짜 기준으로 최근 3년만

    df["Date"] = pd.to_datetime(df["Date"])
    cutoff = df["Date"].max() - relativedelta(years=3)
    df = df[df["Date"] >= cutoff].reset_index(drop=True)

    print("▶ Computing future returns")
    df = compute_future_returns(df, days=[1,2,3,4,5])

    print("▶ Segmenting and summarizing")
    seg_summary = segment_and_summarize(df, days=[1,2,3,4,5])

    ordered_cols = [
        "Stage","day","indicator","segment_lo","segment_hi",
        "total_count","pos_match","pos_ratio",
        "mean_return","neg3_count","neg3_ratio","min_return"
    ]
    seg_summary = (
        seg_summary[ordered_cols]
        .sort_values(["Stage","day","indicator"])
        .reset_index(drop=True)
    )

    seg_summary.to_csv(OUT_CSV, index=False, encoding='utf-8-sig')
    print("▶ Finished. Results saved to", OUT_CSV)

if __name__ == "__main__":
    main()