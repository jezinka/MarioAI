import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

from callback import TrainAndLoggingCallback


def create_env():
    env = gym_super_mario_bros.make('SuperMarioBros-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env, keep_dim=True)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, 4, channels_order='last')
    return env


def create_model(model_env):
    # AI model started
    return PPO('CnnPolicy', model_env, verbose=1, tensorboard_log='./logs/', learning_rate=0.000001, n_steps=512)


def learn_model(model):
    model.learn(total_timesteps=1000000, callback=TrainAndLoggingCallback(check_freq=10000, save_path='./train/'))


if __name__ == '__main__':
    play_env = create_env()
    model = create_model(play_env)
    model.load('./train/best_model_40000')
    state = play_env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = play_env.step(action)
        play_env.render()
