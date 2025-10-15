import numpy as np
import pandas as pd
import cvxpy as cp
import gymnasium as gym 
from stable_baselines3 import DQN, PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from bayes_opt import BayesianOptimization

# 1) 동적 룰 베이스 워밍업
def get_initial_weights(df, method='inv_vol', vol_col='vol_ma5', sharpe_window=12):
    """
    - inv_vol: 1/volatility
    - sharpe: 과거 샤프비율 기반 가중치
    """
    codes = df['Code'].unique()
    if method == 'inv_vol':
        vols = df.groupby('Code')[vol_col].last()
        w = 1 / vols.clip(lower=1e-3)
    else:  # sharpe
        rets = df.groupby('Code')['ret1'].rolling(sharpe_window).apply(lambda x: x.mean()/x.std()).reset_index(0,drop=True)
        sharpe = rets.groupby(df['Code']).last()
        w = sharpe.clip(lower=0)
    w = w / w.sum()
    return w.values  # 종목별 초기 가중치 벡터

# 2) 강건 최적화 함수
def robust_optimize_weights(w_rl, VaR_limit=0.05, w_min=0.0, w_max=1.0, cov=None, alpha=0.95):
    """
    RL 에이전트가 제안한 w_rl에 추가 제약을 걸어 최종 가중치 산출
    - ∑w_i=1, w_min≤w_i≤w_max
    - CVaR/VaR 페널티 (sigma² 대신 quad_form 예시)
    """
    n = len(w_rl)
    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        w <= w_max
    ]
    # 포트폴리오 분산 페널티
    if cov is not None:
        portfolio_var = cp.quad_form(w, cov)
    else:
        portfolio_var = 0

    obj = cp.Minimize(cp.sum_squares(w - w_rl) + 10 * portfolio_var)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return w.value

# 3) 강화학습 환경 정의
class PortfolioEnv(gym.Env):
    """
    state: 정규화된 팩터 스코어 행렬 (n_assets × n_factors)
    action: 자산별 가중치 (합=1)
    reward: 다음 기간 포트폴리오 수익 − λ * 포트폴리오 변동성
    """
    def __init__(self, data, factor_cols, rebalance_dates, risk_lambda=0.5):
        super().__init__()
        self.data = data
        self.factors = factor_cols
        self.dates = rebalance_dates
        self.risk_lambda = risk_lambda
        self.n = len(data['Code'].unique())
        self.current_step = 0

        # action: n assets, sum to 1
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.n,), dtype=np.float32)
        # state: n_assets × n_factors 스택을 1D로
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.n*len(factor_cols),), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        # let Gymnasium do its own seeding
        super().reset(seed=seed)
        # reset your state
        self.current_step = 0
        obs = self._get_state()
        # return obs plus empty info dict
        return obs, {}

    def _get_state(self):
        dt = self.dates[self.current_step]
        df = self.data[self.data['date']==dt].sort_values('Code')
        X = df[self.factors].values  # (n, f)
        return X.flatten()

    def step(self, action):
        # 가중치 정규화
        w = np.clip(action, 0, 1)
        w = w / (w.sum()+1e-8)
        
        # 다음 달 수익 & 리스크
        dt_next = self.dates[self.current_step+1]
        df_now  = self.data[self.data['date']==self.dates[self.current_step]].sort_values('Code')
        df_next = self.data[self.data['date']==dt_next].sort_values('Code')
        rets    = (df_next['close'].values / df_now['close'].values) - 1
        
        port_ret = np.dot(w, rets)
        port_vol = np.std(rets * w)
        reward = port_ret - self.risk_lambda * port_vol
        
        self.current_step += 1
        done = (self.current_step >= len(self.dates)-1)
        return self._get_state(), reward, done, {}

# 4) 베이지안 최적화: RL 하이퍼파라미터 튜닝
def optimize_hyperparams(env_fn, init_params, pbounds, n_iter=20):
    def train_and_eval(**params):
        # ① env 전용 파라미터만 꺼내 환경 생성
        risk = params.pop('risk_lambda')
        env = DummyVecEnv([lambda: env_fn({'risk_lambda': risk})])

        # ② 모델 생성자에 허용된 키만 넘기기
        allowed = ['learning_rate', 'gamma']
        model_kwargs = {k: params[k] for k in allowed if k in params}

        model = init_params['algo'](
            'MlpPolicy',
            env,
            **model_kwargs
        )
        model.learn(total_timesteps=init_params['timesteps'])

        # ③ 성능 평가 (예: 평균 에피소드 reward)
        return float(np.mean(model.ep_info_buffer))

    optimizer = BayesianOptimization(
        f=train_and_eval,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=n_iter)
    return optimizer.max['params']

# 5) 앙상블: 서로 다른 RL 알고리즘으로 가중치 제안 → 평균
def ensemble_weights(env, trained_models, cov):
    w_list = []
    for model in trained_models:
        obs = env.reset()
        action, _ = model.predict(obs)
        w_list.append(action)
    # 평균 가중치
    w_ens = np.mean(np.vstack(w_list), axis=0)
    # robust optimization 적용 예시
    w_final = robust_optimize_weights(w_ens, cov=cov)
    return w_final

# ======================
# 메인 실행 예시
# ======================
if __name__ == '__main__':
    # 1) 데이터 준비 (예시)
    df = pd.read_parquet('data.Analysis.parquet')
    # "date", "Code", "close", "vol5", "ret1", ... 컬럼 필요
    rebalance_dates = sorted(df['date'].dt.to_period('M').dt.to_timestamp('M').unique())
    factor_cols = ['market_cap_z','turnover5_z','vol_ma5_z','mom1m_z']

    # 2) 초기 가중치 (Warmup)
    w0 = get_initial_weights(df, method='inv_vol')

    # 3) RL 환경 생성
    env_fn = lambda params: PortfolioEnv(df, factor_cols, rebalance_dates, risk_lambda=params['risk_lambda'])
    
    # 4) 하이퍼파라미터 최적화 (Bayesian)
    pbounds = {
        'learning_rate': (1e-5, 1e-3),
        'gamma': (0.9, 0.999),
        'risk_lambda': (0.1, 1.0)
    }
    best_params = optimize_hyperparams(env_fn,
                                       init_params={'algo':PPO, 'timesteps':100_000},
                                       pbounds=pbounds,
                                       n_iter=10)

    # 5) RL 모델 학습 (DQN, PPO, A2C)
    models = []
    for Algo in [DQN, PPO, A2C]:
        env_vec = DummyVecEnv([lambda: env_fn(best_params)])
        model = Algo('MlpPolicy', env_vec, **best_params)
        model.learn(total_timesteps=200_000)
        models.append(model)

    # 6) 앙상블 & 강건 최적화
    cov_matrix = df.pivot_table(index='date', columns='Code', values='ret1').cov().values
    env = PortfolioEnv(df, factor_cols, rebalance_dates, risk_lambda=best_params['risk_lambda'])
    w_final = ensemble_weights(env, models, cov_matrix)

    print("최종 가중치:", w_final)