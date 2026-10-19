from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Recria o ambiente da mesma forma
vec_env = DummyVecEnv([make_env])
vec_env = VecNormalize.load("vecnormalize_semaforo.pkl", vec_env)
vec_env.training = False          # não atualiza estatísticas
vec_env.norm_reward = False       # opcional na avaliação

model = PPO.load("modelo_semaforo_ppo")

obs = vec_env.reset()
for _ in range(720):              # 1 hora de simulação
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = vec_env.step(action)