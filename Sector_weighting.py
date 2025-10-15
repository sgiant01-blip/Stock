import numpy as np
import pandas as pd
import cvxpy as cp
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from bayes_opt import BayesianOptimization


# 1) 동적 룰 베이스 워밍업
def get_initial_weights(df, method='inv_vol', vol_col='vol_ma5', sharpe_window=12):
    codes = df['Code'].unique()
    if method == 'inv_vol':
        vols = df.groupby('Code')[vol_col].last()
        w = 1 / vols.clip(lower=1e-3)
    else:
        rets = (
            df.groupby('Code')['ret1']
              .rolling(sharpe_window)
              .apply(lambda x: x.mean() / x.std())
              .reset_index(0, drop=True)
        )
        sharpe = rets.groupby(df['Code']).last()
        w = sharpe.clip(lower=0)
    w = w / w.sum()
    return w.values

# 2) 강건 최적화 함수
def robust_optimize_weights(w_rl,
                            VaR_limit=0.05,
                            w_min=0.0,
                            w_max=1.0,
                            cov=None,
                            alpha=0.95):
    n = len(w_rl)
    w = cp.Variable(n)
    constraints = [
        cp.sum(w) == 1,
        w >= w_min,
        w <= w_max
    ]
    portfolio_var = cp.quad_form(w, cov) if cov is not None else 0
    obj = cp.Minimize(cp.sum_squares(w - w_rl) + 10 * portfolio_var)
    prob = cp.Problem(obj, constraints)
    prob.solve(solver=cp.SCS, verbose=False)
    return w.value


# 3) 강화학습 환경 정의
class PortfolioEnv(gym.Env):
    """
    state: (n_assets × n_factors) flatten
    action: 자산별 가중치 (합=1)
    reward: 다음 월말 수익 − λ * 월말 리턴의 표준편차
    """
    def __init__(self, data, factor_cols, rebalance_dates, risk_lambda=0.5):
        super().__init__()
        self.data = data
        self.factors = factor_cols
        self.dates = rebalance_dates       # month_end 리스트
        self.risk_lambda = risk_lambda

        # 고정된 순서의 전체 종목코드
        self.codes = sorted(self.data['Code'].unique())
        self.n = len(self.codes)
        self.current_step = 0

        self.action_space = gym.spaces.Box(
            low=0, high=1, shape=(self.n,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.n * len(self.factors),),
            dtype=np.float32
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self._get_state()
        return obs, {}

    def _get_state(self):
        # 이번 리밸런스 날짜 기준 슬라이스
        dt = self.dates[self.current_step]
        df_slice = (
            self.data[self.data['month_end'] == dt]
            .set_index('Code')
            .reindex(self.codes)              # 누락 종목 0으로 채움
        )
        X = df_slice[self.factors].fillna(0).values  # (n_assets, n_factors)
        return X.flatten()

    def step(self, action):
        # 1) 비음수, 합1 정규화
        w = np.clip(action, 0, 1)
        w = w / (w.sum() + 1e-8)

        # 2) 다음 월말 리턴 계산
        dt0 = self.dates[self.current_step]
        dt1 = self.dates[self.current_step + 1]
        df0 = (
            self.data[self.data['month_end'] == dt0]
            .set_index('Code')
            .reindex(self.codes)
        )
        df1 = (
            self.data[self.data['month_end'] == dt1]
            .set_index('Code')
            .reindex(self.codes)
        )
        # 종가 누락은 전월말 종가로 채우거나 0 대체
        close0 = df0['close'].fillna(method='ffill').fillna(0).values
        close1 = df1['close'].fillna(method='ffill').fillna(0).values
        rets = close1 / (close0 + 1e-8) - 1

        # 3) 포트폴리오 리턴·리스크·보상
        port_ret = np.dot(w, rets)
        port_vol = np.std(w * rets)
        reward = port_ret - self.risk_lambda * port_vol

        # 4) 다음 스텝 및 종료 여부
        self.current_step += 1
        done = (self.current_step >= len(self.dates) - 1)
        obs = self._get_state()
        return obs, reward, done, {}


# 4) 베이지안 최적화: RL 하이퍼파라미터 튜닝
def optimize_hyperparams(env_fn, init_params, pbounds, n_iter=20):
    """
    env_fn: dict({'risk_lambda':float}) → PortfolioEnv
    init_params: {'algo': PPO/A2C/DQN, 'timesteps': int}
    pbounds: {'learning_rate':(...), 'gamma':(...), 'risk_lambda':(...)}
    """
    def train_and_eval(**params):
        # env-only 파라미터 분리
        risk = params.pop('risk_lambda')
        env = DummyVecEnv([lambda: env_fn({'risk_lambda': risk})])

        # 모델 생성자 허용 키만 전달
        allowed = ['learning_rate', 'gamma']
        model_kwargs = {k: params[k] for k in allowed if k in params}

        model = init_params['algo']('MlpPolicy', env, **model_kwargs)
        model.learn(total_timesteps=init_params['timesteps'])

        # 평균 에피소드 리워드
        return float(np.mean(model.ep_info_buffer))

    optimizer = BayesianOptimization(
        f=train_and_eval,
        pbounds=pbounds,
        random_state=42
    )
    optimizer.maximize(init_points=5, n_iter=n_iter)
    return optimizer.max['params']


# 5) 앙상블: 여러 RL 모델 제안 가중치 평균 → 강건 최적화
def ensemble_weights(env, trained_models, cov):
    w_list = []
    for model in trained_models:
        obs, _ = env.reset()
        action, _ = model.predict(obs)
        w_list.append(action)
    w_ens = np.mean(np.vstack(w_list), axis=0)
    return robust_optimize_weights(w_ens, cov=cov)
