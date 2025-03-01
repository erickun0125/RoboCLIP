import os
import cv2
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from moviepy.editor import ImageSequenceClip

from Metaworld import metaworld
from Metaworld.metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE,
                            ALL_V2_ENVIRONMENTS_GOAL_HIDDEN)

from metaworld_envs import MetaworldSparse

# 학습된 모델을 평가하고 시각화하는 함수
def evaluate_and_render(model_path, env_type, env_id, num_episodes=5, output_video='output.mp4'):
    # 환경 생성
    env = MetaworldSparse(env_id=env_id, text_string="robot opening door", time=True, rank=0)
    env = DummyVecEnv([lambda: MetaworldSparse(env_id=env_id, text_string="robot opening door", time=True, rank=0)])
    model = PPO.load(model_path, env=env)
    
    frames = []  # 저장할 프레임 리스트
    total_rewards = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            
            # 렌더링된 프레임 저장
            frame = env.envs[0].render()
            frames.append(frame)
            
            episode_reward += reward
            
        total_rewards.append(episode_reward)
        print(f"Episode {episode+1}: Reward = {episode_reward}")
    
    env.close()
    
    # 비디오 저장
    clip = ImageSequenceClip([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames], fps=30)
    clip.write_videofile(output_video, codec='libx264')
    print(f"Video saved as {output_video}")
    
    return total_rewards

if __name__ == "__main__":
    model_path = "metaworld/door-open-v2-goal-hidden_sparse_learnt_exp_1/best_model.zip"
    env_id = "door-open-v2-goal-hidden"
    env_type = "sparse_learnt"
    output_video = "ppo_evaluation.mp4"
    
    rewards = evaluate_and_render(model_path, env_type, env_id, num_episodes=5, output_video=output_video)
    print("Evaluation completed. Rewards:", rewards)
