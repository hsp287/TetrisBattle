import os
import random
import pygame
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from video_recorder import VideoRecorder
from TetrisBattle.settings import *

def simple_agent():
    # Initialize the environment and video recorder
    env = TetrisDoubleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    recorder = VideoRecorder(save_dir="training")

    recorder.start_recording()  # Start recording
    n_ep = 1

    for i in range(n_ep):
        # Reset the environment
        state = env.reset()
        done = False

        step1 = 0
        step2 = 0
        action1 = 0
        action2 = 0

        # Access the Tetris instance
        tetris1 = env.game_interface.tetris_list[0]["tetris"]
        tetris2 = env.game_interface.tetris_list[1]["tetris"]

        # Get all possible states
        final_states, action_sequences, was_held, rewards = tetris1.get_all_possible_states()

        # Find the maximum reward
        max_reward = max(rewards)

        # Get indices of all states with the maximum reward
        best_indices = [i for i, reward in enumerate(rewards) if reward == max_reward]

        # Choose one randomly if there are multiple
        chosen_index = random.choice(best_indices)

        # Get the corresponding action sequence
        best_action_sequence1 = action_sequences[chosen_index]
        #print(best_action_sequence)

        # Get all possible states
        final_states, action_sequences, was_held, rewards = tetris2.get_all_possible_states()

        # Find the maximum reward
        max_reward = max(rewards)

        # Get indices of all states with the maximum reward
        best_indices = [i for i, reward in enumerate(rewards) if reward == max_reward]

        # Choose one randomly if there are multiple
        chosen_index = random.choice(best_indices)

        # Get the corresponding action sequence
        best_action_sequence2 = action_sequences[chosen_index]

        while not done:
            if step1 >= len(best_action_sequence1):
                # Access the Tetris instance
                tetris1 = env.game_interface.tetris_list[0]["tetris"]

                # Get all possible states
                final_states, action_sequences, was_held, rewards = tetris1.get_all_possible_states()

                # Find the maximum reward
                max_reward = max(rewards)

                # Get indices of all states with the maximum reward
                best_indices = [i for i, reward in enumerate(rewards) if reward == max_reward]

                # Choose one randomly if there are multiple
                chosen_index = random.choice(best_indices)

                # Get the corresponding action sequence
                best_action_sequence1 = action_sequences[chosen_index]
                #print(final_states[chosen_index])
                #print(best_action_sequence)
                step1 = 0

            if step2 >= len(best_action_sequence2):
                # Access the Tetris instance
                tetris2 = env.game_interface.tetris_list[1]["tetris"]

                # Get all possible states
                final_states, action_sequences, was_held, rewards = tetris2.get_all_possible_states()

                # Find the maximum reward
                max_reward = max(rewards)

                # Get indices of all states with the maximum reward
                best_indices = [i for i, reward in enumerate(rewards) if reward == max_reward]

                # Choose one randomly if there are multiple
                chosen_index = random.choice(best_indices)

                # Get the corresponding action sequence
                best_action_sequence2 = action_sequences[chosen_index]
                #print(final_states[chosen_index])
                #print(best_action_sequence)
                step2 = 0

            # Get the next action
            action1 = best_action_sequence1[step1]
            action2 = best_action_sequence2[step2]

            # Check if the action is the same as the last action
            if action1 == 5 or action1 == 6:
                if tetris1.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
                    step1 += 1
            elif action1 == 3 or action1 == 4:
                if tetris1.LAST_ROTATE_TIME >= ROTATE_FREQ:
                    step1 += 1
            else:
                step1 += 1
            
            # Check if the action is the same as the last action
            if action2 == 5 or action2 == 6:
                if tetris2.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
                    step2 += 1
            elif action2 == 3 or action2 == 4:
                if tetris2.LAST_ROTATE_TIME >= ROTATE_FREQ:
                    step2 += 1
            else:
                step2 += 1

            actions = [action1, action2]
            state, reward, done, info = env.step(actions)

            # Capture the current frame
            recorder.capture_frame(env.game_interface.screen)

    print("Episode finished.")
    recorder.stop_and_save("two_agent_episode")

if __name__ == "__main__":
    simple_agent()