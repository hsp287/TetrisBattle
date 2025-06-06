import os
import pygame
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from video_recorder import VideoRecorder

custom_action_sequence = [0, 1, 3, 5, 5, 5, 5, 2]

def test_video_recorder():
    # Initialize the environment and video recorder
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    recorder = VideoRecorder(save_dir="training")

    num_episodes = 1  # Number of episodes to record
    max_steps_per_episode = 1000  # Maximum steps per episode
    action_interval = 10

    recorder.start_recording()  # Start recording

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        action = 0
        frame_counter = 0
        last_action = None

        while not done and step < max_steps_per_episode:
            frame_counter += 1

            # Get the next action
            action = custom_action_sequence[step % len(custom_action_sequence)]

            # Check if the action is the same as the last action
            if action == last_action:
                # Wait for the action interval if the action is the same
                if frame_counter % action_interval == 0:
                    step += 1
            else:
                # Execute the action immediately if it's different
                step += 1
                frame_counter = 0  # Reset the frame counter for different actions

            state, reward, done, info = env.step(action)
            #print(state["grid"][1])  # print grid state for player 1
            # Update the last action
            last_action = action

            # Capture the current frame
            recorder.capture_frame(env.game_interface.screen)

        print(f"Episode {episode + 1} finished.")

    # Stop recording and save the video
    recorder.stop_and_save("test_random_actions")


def test_random_actions():
    # Initialize the environment and video recorder
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    recorder = VideoRecorder(save_dir="training")

    num_episodes = 1  # Number of episodes to record
    max_steps_per_episode = 1000  # Maximum steps per episode

    recorder.start_recording()  # Start recording

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        step = 0
        action = 0

        while not done and step < max_steps_per_episode:
            # Get the next action
            action = env.random_action()

            state, reward, done, info = env.step(action)

            # Capture the current frame
            recorder.capture_frame(env.game_interface.screen)

        print(f"Episode {episode + 1} finished.")

    # Stop recording and save the video
    recorder.stop_and_save("test_random_actions")


def test_possible_states():
    # Initialize the environment
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    state = env.reset()

    # Access the Tetris instance
    tetris = env.game_interface.tetris_list[0]["tetris"]

    # Get all possible states
    final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

    # Print the results
    print(f"Number of possible placements: {len(final_states)}")
    for i, (grid, actions, held, reward) in enumerate(zip(final_states, action_sequences, was_held, rewards)):
        print(f"Placement {i + 1}:")
        print(f"Grid:\n{grid}")
        print(f"Actions: {actions}")
        print(f"Was Held: {held}")
        print(f"Reward: {reward}")
        print("-" * 50)

if __name__ == "__main__":
    test_random_actions()