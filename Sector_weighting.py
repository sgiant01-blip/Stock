import numpy as np
import pandas as pd
import cvxpy as cp
import gymnasium as gym
from stable_baselines3.common.vec_env import DummyVecEnv
from bayes_opt import BayesianOptimization

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

def robust_optimize_weights(w_rl, VaR_limit=0.05, w_min=0.0, w_max=1.0, cov=None, alpha=0.95):
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

class PortfolioEnv(gym.Env):
    """
    월말별 alive_codes 기준으로 observation/action space 동적으로 관리
    """
    def __init__(self, data, factor_cols, rebalance_dates, alive_codes_per_month, **kwargs):
        super().__init__()        
        self.data = data
        self.factors = factor_cols
        self.dates = rebalance_dates
        self.alive_codes_per_month = alive_codes_per_month
        self.risk_lambda = kwargs.get('risk_lambda', 0.5)
        self.current_step = 0
        
        # 최대 종목 수로 고정
        self.max_codes = max(len(codes) for codes in alive_codes_per_month.values())
        print(f"최대 종목 수: {self.max_codes}")
        
        self.codes = self.alive_codes_per_month[self.dates[self.current_step]]
        self.n = len(self.codes)
        
        # Space를 최대 크기로 고정
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.max_codes,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, 
                                                shape=(self.max_codes * len(self.factors),), 
                                                dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.codes = self.alive_codes_per_month[self.dates[self.current_step]]
        self.n = len(self.codes)
        obs = self._get_state()
        return obs, {}

    def _get_state(self):
        dt = self.dates[self.current_step]
        codes = self.alive_codes_per_month[dt]
        
        # date == dt로 필터링 (month_end 아님!)
        df = (self.data[(self.data['date'] == dt) & (self.data['Code'].isin(codes))]
              .drop_duplicates(subset='Code', keep='last')
              .sort_values('Code'))
        
        # 최대 크기로 패딩
        arr = np.zeros((self.max_codes, len(self.factors)), dtype=np.float32)
        
        for i, code in enumerate(codes):
            if i >= self.max_codes:
                break
            row = df[df['Code'] == code]
            if not row.empty:
                vals = row[self.factors].values[0]
                vals = np.nan_to_num(vals, nan=0.0, posinf=0.0, neginf=0.0)
                arr[i] = vals
        
        return arr.flatten()

    def step(self, action):
        dt0 = self.dates[self.current_step]
        
        # 마지막 날짜인지 확인
        if self.current_step >= len(self.dates) - 1:
            obs = self._get_state()
            return obs, 0.0, True, False, {}
        
        dt1 = self.dates[self.current_step + 1]
        codes0 = self.alive_codes_per_month[dt0]
        codes1 = self.alive_codes_per_month[dt1]
        
        # shape 맞추기 (교집합)
        common_codes = list(set(codes0) & set(codes1))
        if not common_codes:
            self.current_step += 1
            done = (self.current_step >= len(self.dates) - 1)
            truncated = False
            obs = self._get_state()
            return obs, 0.0, done, truncated, {}
        
        # date == dt로 필터링 (month_end 아님!)
        df0 = (self.data[(self.data['date'] == dt0) & (self.data['Code'].isin(common_codes))]
               .drop_duplicates(subset='Code', keep='last')
               .set_index('Code')
               .reindex(common_codes))
        
        df1 = (self.data[(self.data['date'] == dt1) & (self.data['Code'].isin(common_codes))]
               .drop_duplicates(subset='Code', keep='last')
               .set_index('Code')
               .reindex(common_codes))
        
        close0 = df0['close'].ffill().fillna(0).values
        close1 = df1['close'].ffill().fillna(0).values
        
        close0 = np.nan_to_num(close0, nan=1.0, posinf=1.0, neginf=1.0)
        close1 = np.nan_to_num(close1, nan=1.0, posinf=1.0, neginf=1.0)
        
        # action을 실제 종목 수만큼만 사용
        action_slice = action[:len(codes0)]
        w = np.array([action_slice[codes0.index(code)] if code in codes0 else 0.0 
                      for code in common_codes])
        w = np.clip(w, 0, 1)
        w = w / (w.sum() + 1e-8)
        
        rets = close1 / (close0 + 1e-8) - 1
        rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)
        
        port_ret = np.dot(w, rets)
        port_vol = np.std(w * rets) + 1e-8
        reward = port_ret - self.risk_lambda * port_vol
        
        if np.isnan(reward) or np.isinf(reward):
            reward = 0.0
        
        self.current_step += 1
        done = (self.current_step >= len(self.dates) - 1)
        truncated = False
        
        if not done:
            self.codes = self.alive_codes_per_month[self.dates[self.current_step]]
            self.n = len(self.codes)
        
        obs = self._get_state()
        info = {}
        
        return obs, reward, done, truncated, {}

def optimize_hyperparams(env_fn, init_params, pbounds, n_iter=20):
    print("베이지안 최적화 시작...")
    print(f"총 {5 + n_iter}번 시도, 각 {init_params['timesteps']:,} 스텝")
    
    iteration_count = [0]
    
    def train_and_eval(**params):
        iteration_count[0] += 1
        print(f"\n{'='*60}")
        print(f"시도 [{iteration_count[0]}/{5 + n_iter}]")
        print(f"파라미터: lr={params.get('learning_rate', 0):.6f}, "
              f"gamma={params.get('gamma', 0):.4f}, "
              f"risk_lambda={params.get('risk_lambda', 0):.3f}")
        print(f"{'='*60}")
        
        import time
        start_time = time.time()
        
        risk = params.pop('risk_lambda')
        env = DummyVecEnv([lambda: env_fn({'risk_lambda': risk})])
        allowed = ['learning_rate', 'gamma']
        model_kwargs = {k: params[k] for k in allowed if k in params}
        model = init_params['algo']('MlpPolicy', env, verbose=0, **model_kwargs)
        
        print("학습 시작...", flush=True)
        model.learn(total_timesteps=init_params['timesteps'])
        
        elapsed = time.time() - start_time
        
        ep_rewards = [ep_info.get("r", 0) for ep_info in model.ep_info_buffer if "r" in ep_info]
        score = np.mean(ep_rewards) if ep_rewards else 0.0
        
        print(f"✓ 완료! 점수: {score:.4f}, 소요시간: {elapsed:.1f}초")
        return float(score)
    
    optimizer = BayesianOptimization(
        f=train_and_eval,
        pbounds=pbounds,
        random_state=42,
        verbose=2
    )
    
    import time
    total_start = time.time()
    optimizer.maximize(init_points=5, n_iter=n_iter)
    total_elapsed = time.time() - total_start
    
    print(f"\n{'='*60}")
    print(f"최적화 완료!")
    print(f"최대 점수: {optimizer.max['target']:.4f}")
    print(f"총 소요시간: {total_elapsed/60:.1f}분")
    print(f"{'='*60}")
    
    return optimizer.max['params']

def ensemble_weights(env, trained_models, cov, dt, codes):
    w_list = []
    # 환경을 해당 월말 기준으로 초기화
    env.current_step = env.dates.index(dt)
    env.codes = codes
    env.n = len(codes)
    obs, _ = env.reset()
    for model in trained_models:
        action, _ = model.predict(obs)
        w_list.append(action[:len(codes)])  # 실제 종목 수만큼만 사용
    w_ens = np.mean(np.vstack(w_list), axis=0)
    w_final = robust_optimize_weights(w_ens, cov=cov)
    return w_final
