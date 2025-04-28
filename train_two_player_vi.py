import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from TetrisBattle.settings import *
from TetrisBattle.envs.tetris_env import TetrisDoubleEnv
from video_recorder import VideoRecorder
import csv
import random


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
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


def agent_future_state(env, value_net, episode_data, epsilon, state, player):
    # Get all possible states
    tetris = env.game_interface.tetris_list[player-1]["tetris"]
    final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

    # Compute r(s'|s) + V(s') for each possible state
    future_values = []
    for grid, reward in zip(final_states, rewards):
        grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device) 
        # For future states, set attacked to 0
        attacked = 0
        info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0).to(device)
        value = value_net(grid_tensor, info_vector).item()
        future_values.append(reward + value)

    # Select the best future state
    # Epsilon-greedy selection
    if random.random() < epsilon:  # Exploration
        best_index = random.randint(0, len(final_states) - 1)
    else:  # Exploitation
        best_index = future_values.index(max(future_values))
    best_action_sequence = action_sequences[best_index]
    best_future_state = final_states[best_index]
    best_reward = rewards[best_index]
    
    # Include the current state's attacked value in the episode data
    current_attacked = 1 if tetris.attacked > 0 else 0
    episode_data.append((state["grid"][player], current_attacked, best_reward, best_future_state))
    return best_action_sequence, episode_data, best_reward


def check_action_time(tetris, action, step):
    if action == 5 or action == 6:
        if tetris.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ:
            step += 1
    elif action == 3 or action == 4:
        if tetris.LAST_ROTATE_TIME >= ROTATE_FREQ:
            step += 1
    else:
        step += 1    
    return step


def train_agent(num_episodes=100, gamma=0.99, learning_rate=1e-3, record_interval=10, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.95, device="cpu"):
    # Initialize environment, value network, optimizer, and video recorder
    env = TetrisDoubleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    value_net1 = ValueNetwork().to(device)
    optimizer1 = optim.Adam(value_net1.parameters(), lr=learning_rate)
    value_net2 = ValueNetwork().to(device)
    optimizer2 = optim.Adam(value_net2.parameters(), lr=learning_rate)
    recorder = VideoRecorder(save_dir="training_videos")
    best_model_path1 = "best_model_agent1.pth"
    best_model_path2 = "best_model_agent2.pth"
    best_1 = -float('inf')
    best_2 = -float('inf')
    best_lines_sent1 = -float('inf')  # Track the best total lines sent by Agent 1
    best_lines_sent2 = -float('inf')  # Track the best total lines sent by Agent 2

    # Initialize CSV file
    csv_file = "episodic_rewards.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Average Reward1", "Average Loss1", "Total Lines sent1", "Average Reward2", "Average Loss2", "Total Lines sent2", "Winner"])  # Write the header

    # Initialize epsilon
    epsilon = epsilon_start

    recorder.start_recording()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_sent1 = 0  # Track `tetris.sent` for this episode
        step1 = 0
        ep_return1 = 0
        episode_data1 = []   
        count1 = 0
        ep_loss1 = 0
        total_sent2 = 0  # Track `tetris.sent` for this episode
        step2 = 0
        ep_return2 = 0
        episode_data2 = []   
        count2 = 0
        ep_loss2 = 0

        # get initial action sequence
        # Access the Tetris instance
        best_action_sequence1, episode_data1, best_reward1 = agent_future_state(env, value_net1, episode_data1, epsilon, state, player=1)
        ep_return1 += best_reward1
        best_action_sequence2, episode_data2, best_reward2 = agent_future_state(env, value_net2, episode_data2, epsilon, state, player=2)
        ep_return2 += best_reward2

        while not done:
            if step1 >= len(best_action_sequence1):
                # Get all possible states for agent 1
                best_action_sequence1, episode_data1, best_reward1 = agent_future_state(env, value_net1, episode_data1, epsilon, state, player=1)
                ep_return1 += best_reward1
                step1 = 0
                count1 += 1
            
            if step2 >= len(best_action_sequence2):
                # Get all possible states for agent 2
                best_action_sequence2, episode_data2, best_reward2 = agent_future_state(env, value_net2, episode_data2, epsilon, state, player=2)
                ep_return2 += best_reward2
                step2 = 0
                count2 += 1

            # Execute the action sequence
            # Get the next action
            action1 = best_action_sequence1[step1]
            action2 = best_action_sequence2[step2]

            # Check if the action is the same as the last action
            tetris1 = env.game_interface.tetris_list[0]["tetris"]
            tetris2 = env.game_interface.tetris_list[1]["tetris"]
            step1 = check_action_time(tetris1, action1, step1)
            step2 = check_action_time(tetris2, action2, step2)

            action = [action1, action2]
            
            env.now_player = 0
            state, reward, done, info = env.step(action)
            total_sent1 = tetris1.sent
            total_sent2 = tetris2.sent
            recorder.capture_frame(env.game_interface.screen)
        
        # give penalty to reward for losing agent in data
        if info['winner'] == 0:
            last_state, attacked, last_reward, last_future_state = episode_data2[-1]
            updated_reward = last_reward - 100
            episode_data2[-1] = (last_state, attacked, updated_reward, last_future_state)
            ep_return2 += -100
        else:
            last_state, attacked, last_reward, last_future_state = episode_data1[-1]
            updated_reward = last_reward - 100
            episode_data1[-1] = (last_state, attacked, updated_reward, last_future_state)
            ep_return1 += -100

        # update the value networks
        if info['winner'] == 0:  # Agent 1 wins
            # Update Agent 2 using Agent 1's value network
            for s, attacked, r, s_prime in episode_data2:
                # Convert states to tensors
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # Include the attacked value in the info vector
                info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0).to(device)
                future_info_vector = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)  # Future attacked is 0

                # Use Agent 1's value network to guide Agent 2
                with torch.no_grad():
                    v_s_prime_winner = value_net1(s_prime_tensor, future_info_vector).item()  # Agent 1's prediction
                target = gamma * (r + v_s_prime_winner)

                # Compute loss
                v_s = value_net2(s_tensor, info_vector)
                loss = nn.MSELoss()(v_s, torch.tensor([[target]], dtype=torch.float32).to(device))
                ep_loss2 += loss.item()
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

            # Perform a soft update to blend Agent 2's network with Agent 1's network
            tau = 0.2  # Blending factor
            for target_param, param in zip(value_net2.parameters(), value_net1.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # update agent 1 using its own data
            for s, attacked, r, s_prime in episode_data1:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0).to(device)
                future_info_vector = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    v_s_prime = value_net1(s_prime_tensor, future_info_vector).item()
                target = gamma * (r + v_s_prime)
                v_s = value_net1(s_tensor, info_vector)
                loss = nn.MSELoss()(v_s, torch.tensor([[target]], dtype=torch.float32).to(device))
                ep_loss1 += loss.item()
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

        else:  # Agent 2 wins
            # Update Agent 1 using Agent 2's value network
            for s, attacked, r, s_prime in episode_data1:
                # Convert states to tensors
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

                # Include the attacked value in the info vector
                info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0).to(device)
                future_info_vector = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)  # Future attacked is 0

                # Use Agent 2's value network to guide Agent 1
                with torch.no_grad():
                    v_s_prime_winner = value_net2(s_prime_tensor, future_info_vector).item()  # Agent 2's prediction
                target = gamma * (r + v_s_prime_winner)

                # Compute loss
                v_s = value_net1(s_tensor, info_vector)
                loss = nn.MSELoss()(v_s, torch.tensor([[target]], dtype=torch.float32).to(device))
                ep_loss1 += loss.item()
                optimizer1.zero_grad()
                loss.backward()
                optimizer1.step()

            # Perform a soft update to blend Agent 1's network with Agent 2's network
            tau = 0.2  # Blending factor
            for target_param, param in zip(value_net1.parameters(), value_net2.parameters()):
                target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

            # update agent 2 using its own data
            for s, attacked, r, s_prime in episode_data2:
                s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
                info_vector = torch.tensor([attacked], dtype=torch.float32).unsqueeze(0).to(device)
                future_info_vector = torch.tensor([0], dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    v_s_prime = value_net2(s_prime_tensor, future_info_vector).item()
                target = gamma * (r + v_s_prime)
                v_s = value_net2(s_tensor, info_vector)
                loss = nn.MSELoss()(v_s, torch.tensor([[target]], dtype=torch.float32).to(device))
                ep_loss2 += loss.item()
                optimizer2.zero_grad()
                loss.backward()
                optimizer2.step()

        # Log the total reward and total lines sent
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode + 1, ep_return1/count1, ep_loss1/count1, total_sent1, ep_return2/count2, ep_loss2/count2, total_sent2, info['winner']])  # Write episode number, total reward, and total lines sent

        # Print the total reward and total lines sent
        print(f"Episode {episode + 1}: Total Lines Sent1 = {total_sent1}, Total Lines Sent2 = {total_sent2}, Winner = {info['winner']}")

        if total_sent1 > best_lines_sent1:
            best_lines_sent1 = total_sent1
            best_1 = ep_return1
            torch.save(value_net1.state_dict(), best_model_path1)
            print(f"New best model for Agent 1 saved with reward = {best_1/count1}")

        if total_sent2 > best_lines_sent2:
            best_lines_sent2 = total_sent2
            best_2 = ep_return2
            torch.save(value_net2.state_dict(), best_model_path2)
            print(f"New best model for Agent 2 saved with reward = {best_2/count2}")

        # Stop recording and save the video
        if (episode + 1) % record_interval == 0:
            recorder.stop_and_save(f"episode_{episode + 1}")
            recorder.start_recording()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)
    
    print("Training complete.")

if __name__ == "__main__":
    # Check if GPU is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_agent(num_episodes=1000, gamma=0.95, learning_rate=1e-3, record_interval=10, epsilon_start=1.0, epsilon_end=0.00, epsilon_decay=0.985, device=device)