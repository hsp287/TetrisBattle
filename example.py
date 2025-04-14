import os
import pygame
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from video_recorder import VideoRecorder
from agent import Agent

def test_video_recorder():
    # Initialize the environment and video recorder
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    recorder = VideoRecorder(save_dir="training")

    num_episodes = 4  # Number of episodes to record
    max_steps_per_episode = 1000  # Maximum steps per episode

    recorder.start_recording()  # Start recording

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0

        while not done and step < max_steps_per_episode:
            action = env.random_action()  # Take a random action
            state, reward, done, info = env.step(action)

            # Capture the current frame
            recorder.capture_frame(env.game_interface.screen)

            step += 1

        print(f"Episode {episode + 1} finished.")

    # Stop recording and save the video
    recorder.stop_and_save("test_random_actions")

if __name__ == "__main__":
    test_video_recorder()