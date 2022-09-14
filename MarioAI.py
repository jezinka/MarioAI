import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv


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


def continue_learning():
    model_path = f"train/10_multi_model_80000_steps"
    log_path = f"logs/"
    model = PPO.load(model_path, tensorboard_log=log_path)
    num_cpu = 10  # Number of processes to use
    play_env = DummyVecEnv([create_env() for _ in range(num_cpu)])
    play_env = VecFrameStack(play_env, 4, channels_order='last')
    model.set_env(play_env)
    callback = CheckpointCallback(
        save_freq=1000,
        save_path="./train/",
        name_prefix=str(num_cpu) + "_multi_model"
    )
    model.learn(total_timesteps=1000000, callback=callback, reset_num_timesteps=False)


def play_model():
    num_cpu = 1  # Number of processes to use
    play_env = DummyVecEnv([create_env() for _ in range(num_cpu)])
    play_env = VecFrameStack(play_env, 4, channels_order='last')
    model = create_model(play_env)
    model.load('./train/10_multi_model_80000_steps')
    state = play_env.reset()
    while True:
        action, _ = model.predict(state)
        state, reward, done, info = play_env.step(action)
        play_env.render()


if __name__ == '__main__':
    # learn_model()
    # continue_learning()
    play_model()
