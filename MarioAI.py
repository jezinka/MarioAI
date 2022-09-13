import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv, SubprocVecEnv

from callback import TrainAndLoggingCallback


def create_env():
    def _init():
        env = gym_super_mario_bros.make('SuperMarioBros-v0')
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env, keep_dim=True)
        return env

    return _init


def create_model(model_env):
    # AI model started
    return PPO('CnnPolicy', model_env, verbose=1, tensorboard_log='./logs/', learning_rate=0.000001, n_steps=512)


def learn_model():
    num_cpu = 10  # Number of processes to use
    play_env = DummyVecEnv([create_env() for _ in range(num_cpu)])
    play_env = VecFrameStack(play_env, 4, channels_order='last')
    model = create_model(play_env)
    callback = CheckpointCallback(
        save_freq=1000,
        save_path="./train/",
        name_prefix=str(num_cpu) + "_multi_model"
    )
    model.learn(total_timesteps=1000000, callback=callback)


def play_model():
    play_env = gym_super_mario_bros.make('SuperMarioBros-v0')
    play_env = JoypadSpace(play_env, SIMPLE_MOVEMENT)
    play_env = GrayScaleObservation(play_env, keep_dim=True)
    play_env = DummyVecEnv([lambda: play_env])
    play_env = VecFrameStack(play_env, 4, channels_order='last')
    model = create_model(play_env)
    model.load('./train/10_multi_model_20000_steps')
    state = play_env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = play_env.step(action)
        play_env.render()


if __name__ == '__main__':
    # learn_model()
    play_model()
