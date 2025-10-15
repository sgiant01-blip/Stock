import sys
import importlib

# 기존 모듈 제거
if 'Sector_weighting' in sys.modules:
    del sys.modules['Sector_weighting']
if 'Sector_selection' in sys.modules:
    del sys.modules['Sector_selection']

import warnings
warnings.filterwarnings('ignore', message='.*Gym has been unmaintained.*')

import pandas as pd
import numpy as np
from tqdm import tqdm

# 이제 import
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
from stable_baselines3.common.callbacks import BaseCallback

# tqdm 콜백 클래스
class TqdmCallback(BaseCallback):
    def __init__(self, total_timesteps, desc="학습 진행"):
        super().__init__()
        self.pbar = None
        self.total_timesteps = total_timesteps
        self.desc = desc
    
    def _on_training_start(self):
        self.pbar = tqdm(total=self.total_timesteps, desc=self.desc)
    
    def _on_step(self):
        if self.pbar:
            self.pbar.update(1)
        return True
    
    def _on_training_end(self):
        if self.pbar:
            self.pbar.close()

# 기존 함수들 - 반드시 유지!
def select_top_stocks(df: pd.DataFrame,
                      top_n: int = 10,
                      min_volume: float = 1e4,
                      min_turnover: float = 1e6) -> pd.DataFrame:
    result = []
    for dt in sorted(df['month_end'].unique()):
        uni = df[df['month_end'] == dt].copy()
        uni = uni[(uni['volume'] > min_volume) & (uni['trading value'] > min_turnover)]
        alive = (df[(df['Code'].isin(uni['Code'])) & (df['date'] > dt)].groupby('Code')['date'].min() > dt)
        uni = uni[uni['Code'].isin(alive[alive].index)]
        
        # 섹터별 상위 N개 종목 선택
        selected = (
            uni.sort_values('score', ascending=False)
               .groupby('Sector', group_keys=False)
               .head(top_n)
        )
        
        selected['month_end'] = dt
        result.append(selected)
    
    result = [df for df in result if not df.empty]
    if result:
        return pd.concat(result).reset_index(drop=True)
    else:
        return pd.DataFrame()

def calc_next_month_return(df: pd.DataFrame, codes: list, dt: pd.Timestamp) -> np.ndarray:
    curr = df[(df['month_end'] == dt) & (df['Code'].isin(codes))].set_index('Code')['close']
    next_dt = dt + pd.offsets.MonthEnd(1)
    fut = df[(df['date'] == next_dt) & (df['Code'].isin(codes))].set_index('Code')['close']
    rets = (fut / curr - 1).reindex(codes).fillna(0).values
    return rets

def get_alive_codes_per_month(df):
    result = {}
    for dt in sorted(df['month_end'].unique()):
        # 중복 제거하여 유니크한 종목 코드만 가져오기
        codes = df[df['month_end'] == dt].drop_duplicates(subset='Code')['Code'].unique()
        result[dt] = sorted(codes)
    return result

def main():
    # 로드된 모듈 파일 경로 확인
    import Sector_weighting
    print(f"Sector_weighting 로드 위치: {Sector_weighting.__file__}")
    
    # step 메서드 소스 확인
    import inspect
    step_source = inspect.getsource(Sector_weighting.PortfolioEnv.step)
    if '.drop_duplicates' in step_source:
        print("✓ step 메서드가 올바르게 로드되었습니다.")
    else:
        print("✗ 경고: step 메서드가 구버전입니다!")
        
    # 1) 데이터 및 팩터 준비
    print("=" * 60)
    print("1단계: 데이터 로드 및 전처리")
    print("=" * 60)
    
    df = load_data('Sample.data.Analysis.50.parquet')
    df = preprocess_data(df)
    
    # 중복 데이터 제거
    print(f"중복 제거 전 데이터 수: {len(df):,}")
    df = df.drop_duplicates(subset=['date', 'Code'], keep='last')
    print(f"중복 제거 후 데이터 수: {len(df):,}")
    
    print("\n특성 엔지니어링 중...")
    df = feature_engineering(df)
    df = winsorize_features(df, ['market_cap','turnover5','vol_ma5','mom1m'])
    df = standardize_and_score(df, ['market_cap','turnover5','vol_ma5','mom1m'])
    df = apply_inverse_vol_weight(df)
    
    # NaN 처리
    factor_cols = ['market_cap_z','turnover5_z','vol_ma5_z','mom1m_z']
    print(f"\nNaN count before filtering:\n{df[factor_cols].isnull().sum()}")
    df[factor_cols] = df[factor_cols].fillna(0)
    df['score'] = df['score'].fillna(0)
    df['score_raw'] = df['score_raw'].fillna(0)
    print(f"NaN count after filtering:\n{df[factor_cols].isnull().sum()}")
    
    rebalance_dates = sorted(df['month_end'].unique())
    print(f"\n리밸런싱 날짜 수: {len(rebalance_dates)}")
    print(f"기간: {rebalance_dates[0]} ~ {rebalance_dates[-1]}")
    
    alive_codes_per_month = get_alive_codes_per_month(df)

    # 2) 리밸런싱 종목 선정
    print("\n" + "=" * 60)
    print("2단계: 종목 선정")
    print("=" * 60)
    
    selected_df = select_top_stocks(df)
    
    if selected_df.empty:
        print("⚠️ 선택된 종목이 없습니다. 프로그램을 종료합니다.")
        return
    
    print(f"선택된 종목 수: {selected_df['Code'].nunique():,}")
    print(f"전체 선택 데이터: {len(selected_df):,} 행")

    # 3) 초기 가중치
    print("\n" + "=" * 60)
    print("3단계: 초기 가중치 계산")
    print("=" * 60)
    
    w0 = get_initial_weights(df, method='inv_vol')
    print(f"초기 가중치 계산 완료 (종목 수: {len(w0)})")

    # 4) RL 환경 및 하이퍼파라미터 최적화
    print("\n" + "=" * 60)
    print("4단계: 하이퍼파라미터 최적화")
    print("=" * 60)
    
    env_fn = lambda p: PortfolioEnv(
        df, factor_cols, rebalance_dates, alive_codes_per_month,
        risk_lambda=p['risk_lambda'],
        **{k: v for k, v in p.items() if k != 'risk_lambda'}
    )

    """
    # 실제 학습용
    best_params = optimize_hyperparams(
        env_fn,
        init_params={'algo': PPO, 'timesteps': 100_000},
        pbounds={'learning_rate': (1e-5, 1e-3),
                 'gamma':         (0.9, 0.999),
                 'risk_lambda':   (0.1, 0.8)},
        n_iter=10
    )
    """
    # 빠른 테스트용
    best_params = optimize_hyperparams(
        env_fn,
        init_params={'algo': PPO, 'timesteps': 5_000},  # 5천 스텝
        pbounds={'learning_rate': (1e-4, 5e-4),
                'gamma':         (0.98, 0.999),
                'risk_lambda':   (0.4, 0.6)},
        n_iter=2  # 총 7번만
    )   

    print(f"\n최적 하이퍼파라미터:")
    for k, v in best_params.items():
        print(f"  {k}: {v:.6f}")

    # 5) RL 모델 학습
    print("\n" + "=" * 60)
    print("5단계: RL 모델 학습")
    print("=" * 60)
    
    models = []
    algos = [PPO, A2C, DQN]
    
    for i, Algo in enumerate(algos, 1):
        print(f"\n[{i}/{len(algos)}] {Algo.__name__} 학습 중...")
        env_vec = DummyVecEnv([lambda: env_fn(best_params)])
        model = Algo('MlpPolicy', env_vec, 
                     learning_rate=best_params.get('learning_rate', 3e-4),
                     gamma=best_params.get('gamma', 0.99),
                     verbose=0)
        
        # tqdm 콜백 사용
        callback = TqdmCallback(total_timesteps=200_000, desc=f"{Algo.__name__} 학습")
        model.learn(total_timesteps=200_000, callback=callback)
        models.append(model)
        print(f"✓ {Algo.__name__} 학습 완료")

    # 6) 백테스트
    print("\n" + "=" * 60)
    print("6단계: 백테스트")
    print("=" * 60)
    
    results = []
    for dt in tqdm(rebalance_dates, desc="백테스트 진행", unit="월"):
        codes = alive_codes_per_month[dt]
        uni = selected_df[selected_df['month_end'] == dt]
        sel_codes = [c for c in codes if c in uni['Code'].tolist()]
        
        if not sel_codes:
            results.append({'date': dt, 'ret': 0})
            continue
        
        # 공분산 행렬 계산
        pivot_ret = df.pivot_table(index='date', columns='Code', values='ret1')
        pivot_ret = pivot_ret.fillna(0)
        
        cov = pivot_ret.cov()
        cov = cov.loc[sel_codes, sel_codes].values
        cov = np.nan_to_num(cov, nan=0.0)
        
        env = PortfolioEnv(df, factor_cols, rebalance_dates, alive_codes_per_month, 
                          risk_lambda=best_params['risk_lambda'])
        
        w_rl = ensemble_weights(env, models, cov, dt, sel_codes)
        w_norm = w_rl / (w_rl.sum() + 1e-8)
        
        ret_next = calc_next_month_return(df, sel_codes, dt)
        port_ret = np.dot(w_norm, ret_next)
        
        results.append({'date': dt, 'ret': port_ret})

    # 결과 분석
    backtest = pd.DataFrame(results).set_index('date')
    backtest['cum_ret'] = (1 + backtest['ret']).cumprod() - 1
    
    print("\n" + "=" * 60)
    print("백테스트 결과")
    print("=" * 60)
    print(backtest.tail(10))
    
    print("\n" + "=" * 60)
    print("성과 지표")
    print("=" * 60)
    print(f"최종 누적 수익률:    {backtest['cum_ret'].iloc[-1]:>10.2%}")
    print(f"연평균 수익률:       {backtest['ret'].mean() * 12:>10.2%}")
    print(f"변동성 (연환산):     {backtest['ret'].std() * np.sqrt(12):>10.2%}")
    
    sharpe = backtest['ret'].mean() / (backtest['ret'].std() + 1e-8) * np.sqrt(12)
    print(f"샤프 비율:          {sharpe:>10.2f}")
    
    max_dd = (backtest['cum_ret'].cummax() - backtest['cum_ret']).max()
    print(f"최대 낙폭 (MDD):     {max_dd:>10.2%}")
    
    # 결과 저장
    backtest.to_csv('backtest_results.csv')
    print(f"\n✓ 결과가 'backtest_results.csv'에 저장되었습니다.")

if __name__ == '__main__':
    main()
