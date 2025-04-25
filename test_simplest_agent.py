import os
import random
import pygame
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from video_recorder import VideoRecorder
from TetrisBattle.settings import *

def simple_agent():
    # Initialize the environment and video recorder
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    recorder = VideoRecorder(save_dir="training")

    recorder.start_recording()  # Start recording
    n_ep = 10

    for i in range(n_ep):
        # Reset the environment
        state = env.reset()
        done = False

        step = 0
        action = 0

        # Access the Tetris instance
        tetris = env.game_interface.tetris_list[0]["tetris"]

        # Get all possible states
        final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

        # Find the maximum reward
        max_reward = max(rewards)

        # Get indices of all states with the maximum reward
        best_indices = [i for i, reward in enumerate(rewards) if reward == max_reward]

        # Choose one randomly if there are multiple
        chosen_index = random.choice(best_indices)

        # Get the corresponding action sequence
        best_action_sequence = action_sequences[chosen_index]
        print(best_action_sequence)

        while not done:
            if step >= len(best_action_sequence):
                # Access the Tetris instance
                tetris = env.game_interface.tetris_list[0]["tetris"]

                # Get all possible states
                final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

                # Find the maximum reward
                max_reward = max(rewards)

                # Get indices of all states with the maximum reward
                best_indices = [i for i, reward in enumerate(rewards) if reward == max_reward]

                # Choose one randomly if there are multiple
                chosen_index = random.choice(best_indices)

                # Get the corresponding action sequence
                best_action_sequence = action_sequences[chosen_index]
                #print(final_states[chosen_index])
                #print(best_action_sequence)
                step = 0

            # Get the next action
            action = best_action_sequence[step]

            # Check if the action is the same as the last action
            if action == 5 or action == 6:
                if tetris.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
                    step += 1
            elif action == 3 or action == 4:
                if tetris.LAST_ROTATE_TIME >= ROTATE_FREQ:
                    step += 1
            else:
                step += 1

            state, reward, done, info = env.step(action)

            # Capture the current frame
            recorder.capture_frame(env.game_interface.screen)

    print("Episode finished.")
    recorder.stop_and_save("simple_agent_episode")

if __name__ == "__main__":
    simple_agent()