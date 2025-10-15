import pandas as pd
import numpy as np
from tqdm import tqdm

from Sector_selection import (
    load_data, preprocess_data, feature_engineering,
    winsorize_features, standardize_and_score,
    apply_inverse_vol_weight
)
from Sector_weighting import (
    get_initial_weights,
    robust_optimize_weights,
    PortfolioEnv, optimize_hyperparams,
    ensemble_weights
)
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv

def select_top_stocks(df: pd.DataFrame,
                      top_n: int = 10,
                      min_volume: float = 1e4,
                      min_turnover: float = 1e6) -> pd.DataFrame:
    # … (기존 구현 그대로) …
    result = []
    for dt in sorted(df['month_end'].unique()):
        uni = df[df['month_end'] == dt].copy()
        uni = uni[(uni['volume'] > min_volume) &
                  (uni['trading value'] > min_turnover)]
        alive = (df[(df['Code'].isin(uni['Code'])) & (df['date'] > dt)]
                 .groupby('Code')['date'].min() > dt)
        uni = uni[uni['Code'].isin(alive[alive].index)]
        selected = uni.groupby('Sector', group_keys=False) \
                      .apply(lambda g: g.nlargest(top_n, 'score'))
        selected['month_end'] = dt
        result.append(selected)
    return pd.concat(result).reset_index(drop=True)

def calc_next_month_return(df: pd.DataFrame,
                           codes: list,
                           dt: pd.Timestamp) -> np.ndarray:
    """
    선택된 종목 리스트(codes)의 다음 월말 수익률 벡터를 반환
    """
    # 현월말 종가
    curr = (df[(df['month_end'] == dt) &
               (df['Code'].isin(codes))]
            .set_index('Code')['close'])
    # 다음 월말 날짜
    next_dt = dt + pd.offsets.MonthEnd(1)
    # 다음 월말 종가
    fut = (df[(df['date'] == next_dt) &
              (df['Code'].isin(codes))]
           .set_index('Code')['close'])
    # 수익률 계산, 순서 맞추고 NaN은 0 대체
    rets = (fut / curr - 1).reindex(codes).fillna(0).values
    return rets


def main():
    # 1) 데이터 및 팩터 준비
    df = load_data('data.Analysis.parquet')
    df = preprocess_data(df)
    df = feature_engineering(df)
    df = winsorize_features(df, ['market_cap','turnover5','vol_ma5','mom1m'])
    df = standardize_and_score(df, ['market_cap','turnover5','vol_ma5','mom1m'])

    # 2) 날짜별 Z-스코어 컬럼 추가
    for col in ['market_cap','turnover5','vol_ma5','mom1m']:
        df[f'{col}_z'] = (
            df.groupby('date')[col]
              .transform(lambda x: (x - x.mean())/(x.std()+1e-8))
        )

    # 2-1) 종합 점수(score) 컬럼 생성
    factor_cols = ['market_cap_z','turnover5_z','vol_ma5_z','mom1m_z']
    df['score'] = df[factor_cols].sum(axis=1)      # 또는 .mean(axis=1)

    # 3) 리밸런싱 종목 선정
    selected_df = select_top_stocks(df)

    # 4) Warmup: 초기 가중치
    w0 = get_initial_weights(df, method='inv_vol')

    # 5) RL 환경 및 하이퍼파라미터 최적화
    rebalance_dates = sorted(df['month_end'].unique())
    factor_cols     = ['market_cap_z','turnover5_z','vol_ma5_z','mom1m_z']
    env_fn = lambda p: PortfolioEnv(
        df, factor_cols, rebalance_dates,
        risk_lambda=p['risk_lambda']
    )

    best_params = optimize_hyperparams(
        env_fn,
        init_params={'algo': PPO, 'timesteps': 100_000},
        pbounds={'learning_rate': (1e-5,1e-3),
                 'gamma':         (0.9,0.999),
                 'risk_lambda':   (0.1,1.0)},
        n_iter=10
    )

    # 6) RL 모델 학습 (DQN, PPO, A2C)
    models = []
    for Algo in [DQN, PPO, A2C]:
        env_vec = DummyVecEnv([lambda: env_fn(best_params)])
        model   = Algo('MlpPolicy', env_vec, **best_params)
        model.learn(total_timesteps=200_000)
        models.append(model)

    # 7) 백테스트 루프: 매월말 포트폴리오 수익 계산
    results = []
    for dt in tqdm(rebalance_dates, desc="백테스트"):
        uni  = selected_df[selected_df['month_end'] == dt]
        codes = uni['Code'].tolist()

        cov   = df.pivot_table(index='date',
                               columns='Code',
                               values='ret1').cov().values
        env   = PortfolioEnv(df, factor_cols, rebalance_dates,
                             risk_lambda=best_params['risk_lambda'])
        w_rl  = ensemble_weights(env, models, cov)

        # 선택 종목만 고르고 재정규화
        w_sel  = pd.Series(w_rl,
                           index=sorted(df['Code'].unique()))[codes]
        w_norm = (w_sel / w_sel.sum()).values

        # 다음 월말 포트폴리오 수익률
        ret_next = calc_next_month_return(df, codes, dt)
        port_ret = np.dot(w_norm, ret_next)
        results.append({'date': dt, 'ret': port_ret})

    backtest = (pd.DataFrame(results)
                  .set_index('date'))
    backtest['cum_ret'] = (1 + backtest['ret']).cumprod() - 1
    print(backtest)


if __name__ == '__main__':
    main()