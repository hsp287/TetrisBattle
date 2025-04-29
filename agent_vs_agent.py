import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from TetrisBattle.settings import *
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from video_recorder import VideoRecorder
import csv
import random
from copy import deepcopy
from TetrisBattle.tetris import get_infos


class MultiAgent(nn.Module):
    def __init__(self):
        super(MultiAgent, self).__init__()
        # Convolutional layers for the grid input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers for the grid input
        self.fc1_grid = nn.Linear(64 * 10 * 20, 128)
        
        # Fully connected layers for the info vector
        self.fc1_vector = nn.Linear(1, 16)
        
        # Final layers combining both inputs
        self.fc2 = nn.Linear(128 + 16, 128)
        self.fc3 = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, grid, vector):
        # Process the grid input
        x_grid = torch.relu(self.conv1(grid))
        x_grid = torch.relu(self.conv2(x_grid))
        x_grid = x_grid.view(x_grid.size(0), -1) 
        x_grid = torch.relu(self.fc1_grid(x_grid))
        
        # Process the vector input
        x_vector = torch.relu(self.fc1_vector(vector))
        
        # Combine both inputs
        x = torch.cat((x_grid, x_vector), dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 10 * 20, 128)
        self.fc2 = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = x.view(x.size(0), -1)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


class ValueNetworkAug(nn.Module):
    def __init__(self):
        super(ValueNetworkAug, self).__init__()
        # Convolutional layers for the grid input
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        
        # Fully connected layers for the grid input
        self.fc1_grid = nn.Linear(64 * 10 * 20, 128)
        
        # Fully connected layers for the info vector
        self.fc1_vector = nn.Linear(7 * 4 + 2, 64)
        
        # Final layers combining both inputs
        self.fc2 = nn.Linear(128 + 64, 128)
        self.fc3 = nn.Linear(128, 1)

        # Initialize weights
        self._initialize_weights()

    def forward(self, grid, vector):
        # Process the grid input
        x_grid = torch.relu(self.conv1(grid))
        x_grid = torch.relu(self.conv2(x_grid))
        x_grid = x_grid.view(x_grid.size(0), -1) 
        x_grid = torch.relu(self.fc1_grid(x_grid))
        
        # Process the vector input
        x_vector = torch.relu(self.fc1_vector(vector))
        
        # Combine both inputs
        x = torch.cat((x_grid, x_vector), dim=1)
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def load_agent(agent_type, model_path):
    if agent_type == "multiagent":
        model = MultiAgent()
    elif agent_type == "valuenetwork":
        model = ValueNetwork()
    elif agent_type == "valuenetworkaug":
        model = ValueNetworkAug()
    elif agent_type == "baseline":
        model = ValueNetwork()
    else:
        raise ValueError("Invalid agent type")
    
    # Load the model weights
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set the model to evaluation mode
    return model


def agent_agent_action(env, value_net, agent_type, state, player):
    # Get all possible states
    tetris = env.game_interface.tetris_list[player-1]["tetris"]
    final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()
    info_state = state["info"][1][:6]

    # Compute r(s'|s) + V(s') for each possible state
    future_values = []
    if agent_type == "multiagent":
        for grid, reward in zip(final_states, rewards):
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            # For future states, set attacked to 0
            attacked = 0
            info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0)
            value = value_net(grid_tensor, info_vector).item()
            future_values.append(reward + value)
    elif agent_type == "valuenetwork":
        for grid, reward in zip(final_states, rewards):
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            value = value_net(grid_tensor).item()
            future_values.append(reward + value) 
    elif agent_type == "valuenetworkaug":
        future_info = deepcopy(info_state[:4])
        for grid, reward in zip(final_states, rewards):
            if was_held:
                if tetris.held is not None:
                    future_info[0] = info_state[2]
                    future_info[1] = info_state[0]
                    future_info[2] = info_state[3]
                    future_info[3] = info_state[4]
                else:
                    future_info[0] = info_state[3]
                    future_info[1] = info_state[0]
                    future_info[2] = info_state[4]
                    future_info[3] = info_state[5]
            else:
                future_info[0] = info_state[2]
                future_info[1] = info_state[1]
                future_info[2] = info_state[3]
                future_info[3] = info_state[4]
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0) 
            height_sum, diff_sum, max_height, holes = get_infos(grid)
            piece_vec = torch.tensor(future_info.flatten(), dtype=torch.float32).unsqueeze(0)
            board_vec = torch.tensor([diff_sum, holes],dtype=torch.float32).unsqueeze(0)
            info_tensor = torch.cat([piece_vec, board_vec], dim=1)
            value = value_net(grid_tensor, info_tensor).item()
            future_values.append(reward + value) 
    else:
        future_values = rewards

    # Select the best future state
    best_index = future_values.index(max(future_values))
    best_action_sequence = action_sequences[best_index]
    
    return best_action_sequence


def agent_versus_agent(agent_type1=None, agent_type2=None, model_path1=None, model_path2=None):
    agent1 = load_agent(agent_type1, model_path1)
    agent2 = load_agent(agent_type2, model_path2)
    # Initialize the environment and video recorder
    env = TetrisDoubleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    recorder = VideoRecorder(save_dir="training")
    recorder.start_recording()  # Start recording

    n_ep = 100

    # Initialize CSV file
    csv_file = "multi_vs_base.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Total Lines sent1", "Total Lines sent2", "Winner"]) 

    for i in range(n_ep):
        # Reset the environment
        state = env.reset()
        done = False

        step1 = 0
        step2 = 0

        # Access the Tetris instance
        tetris1 = env.game_interface.tetris_list[0]["tetris"]
        tetris2 = env.game_interface.tetris_list[1]["tetris"]

        best_action_sequence1 = agent_agent_action(env, agent1, agent_type1, state, player=1)
        best_action_sequence2 = agent_agent_action(env, agent2, agent_type2, state, player=2)

        while not done:
            if step1 >= len(best_action_sequence1):
                best_action_sequence1 = agent_agent_action(env, agent1, agent_type1, state, player=1)
                step1 = 0

            if step2 >= len(best_action_sequence2):
                best_action_sequence2 = agent_agent_action(env, agent2, agent_type2, state, player=2)
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
            total_sent1 = tetris1.sent
            total_sent2 = tetris2.sent

            # Capture the current frame only for every 10th episode
            if (i + 1) % 10 == 1:
                recorder.capture_frame(env.game_interface.screen)

        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([i + 1, total_sent1, total_sent2, info['winner']])  # Write episode number, total reward, and total lines sent

        # Print the total reward and total lines sent
        print(f"Episode {i + 1}: Total Lines Sent1 = {total_sent1}, Total Lines Sent2 = {total_sent2}, Winner = {info['winner']}")

    print("All Episodes finished.")
    recorder.stop_and_save("two_agent_episode")

if __name__ == "__main__":
    agent_versus_agent(agent_type1="multiagent", agent_type2="valuenetwork", model_path1="value_iteration_results/multi/best_model_agent1.pth", model_path2="value_iteration_results/sparse/best_model.pth")