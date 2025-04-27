import torch # type: ignore
import torch.nn as nn # type: ignore
import torch.optim as optim # type: ignore
from TetrisBattle.settings import *
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from video_recorder import VideoRecorder
import csv
import random
from copy import deepcopy
from TetrisBattle.utils.features import *


class ValueNetwork(nn.Module):
    def __init__(self):
        super(ValueNetwork, self).__init__()
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
    

def train_agent(num_episodes=100, gamma=0.99, learning_rate=1e-3, record_interval=10, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.95, device="cpu"):
    # Initialize environment, value network, optimizer, and video recorder
    env = TetrisSingleEnv(gridchoice="none", obs_type="grid", mode="rgb_array")
    value_net = ValueNetwork().to(device)
    optimizer = optim.Adam(value_net.parameters(), lr=learning_rate)
    recorder = VideoRecorder(save_dir="training_videos")

    best_sent = -float('inf')  # Track the highest `tetris.sent`
    best_model_path = "best_model.pth"

    # Initialize CSV file
    csv_file = "episodic_rewards.csv"
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Episode", "Average Reward", "Average Loss", "Total Lines sent"])  # Write the header

    # Initialize epsilon
    epsilon = epsilon_start

    recorder.start_recording()

    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_sent = 0  # Track `tetris.sent` for this episode
        ep_return = 0
        episode_data = []   
        count = 0
        ep_loss = 0

        # Access the Tetris instance
        tetris = env.game_interface.tetris_list[0]["tetris"]
        grid_state = state["grid"][1]
        info_state = state["info"][1][:5]

        while not done:
            # Get all possible states
            final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

            # Compute r(s'|s) + V(s') for each possible state
            future_values = []
            future_infos  = []
            future_info = deepcopy(info_state[:4])
            for grid, reward in zip(final_states, rewards):
                if was_held:
                    future_info[0] = info_state[2]
                    future_info[1] = info_state[0]
                    future_info[2] = info_state[3]
                    future_info[3] = info_state[4]
                else:
                    future_info[0] = info_state[2]
                    future_info[1] = info_state[1]
                    future_info[2] = info_state[3]
                    future_info[3] = info_state[4]
                grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
                height_sum, diff_sum, max_height, holes= tetris.get_infos(grid)
                piece_vec = torch.tensor(future_info.flatten(), dtype=torch.float32).unsqueeze(0).to(device)
                board_vec = torch.tensor([diff_sum, holes],dtype=torch.float32).unsqueeze(0).to(device)
                info_tensor = torch.cat([piece_vec, board_vec], dim=1)
                value = value_net(grid_tensor, piece_vec).item()
                future_values.append(reward + value)
                future_infos.append(piece_vec)

            # Select the best future state
            # Epsilon-greedy selection
            if random.random() < epsilon:  # Exploration
                best_index = random.randint(0, len(final_states) - 1)
            else:  # Exploitation
                best_index = future_values.index(max(future_values))
            best_action_sequence = action_sequences[best_index]
            best_future_state = final_states[best_index]
            best_reward = rewards[best_index]
            best_info   = future_infos[best_index]

            height_sum, diff_sum, max_height, holes= tetris.get_infos(grid_state)
            piece_vec = torch.tensor(info_state[:4].flatten(), dtype=torch.float32).unsqueeze(0).to(device)
            board_vec = torch.tensor([diff_sum, holes],dtype=torch.float32).unsqueeze(0).to(device)
            current_info = torch.cat([piece_vec, board_vec], dim=1)

            episode_data.append((grid_state, current_info, best_reward, best_future_state, best_info))
            ep_return += best_reward
            count += 1

            # rollout chosen action sequence
            step = 0
            while step < len(best_action_sequence) and not done:
                a = best_action_sequence[step]
                if (
                   (a in (5,6) and tetris.LAST_MOVE_SHIFT_TIME > MOVE_SHIFT_FREQ)
                   or
                   (a in (3,4) and tetris.LAST_ROTATE_TIME >= ROTATE_FREQ)
                   or
                   a not in (3,4,5,6)
                ):
                    step += 1

                state, _, done, _ = env.step(a)
                recorder.capture_frame(env.game_interface.screen)
                total_sent = tetris.sent

            # get new current state + info
            if not done:
                grid_state = state["grid"][1]
                info_state = state["info"][1][:5]

        if tetris.check_KO():
            s1, s2, r, s1_prime, s2_prime = episode_data[-1]
            episode_data[-1] = (s1, s2, r-100, s1_prime, s2_prime)
            ep_return += -100

        for s1, s2, r, s1_prime, s2_prime in episode_data:
            # Convert states to tensors
            grid_tensor = torch.tensor(s1, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            info_tensor = s2

            grid_tensor_prime = torch.tensor(s1_prime, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            info_tensor_prime = s2_prime

            # Compute target y = gamma * [r + V(s')]
            with torch.no_grad():
                v_s_prime = value_net(grid_tensor_prime, info_tensor_prime).item()
            target =  r + gamma*v_s_prime

            # Compute loss
            v_s = value_net(grid_tensor, info_tensor)
            loss = nn.MSELoss()(v_s, torch.tensor([[target]], dtype=torch.float32).to(device))
            ep_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            optimizer.step()

        # Log the total reward and total lines sent
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow([episode + 1, ep_return/count, ep_loss/count, total_sent])  # Write episode number, total reward, and total lines sent

        # Print the total reward and total lines sent
        print(f"Episode {episode + 1}: Average Episodic Reward = {ep_return/count}, Average Episodic Loss = {ep_loss/count}, Total Lines Sent = {total_sent}")

        # Save the best model based on `tetris.sent`
        if total_sent > best_sent:
            best_sent = total_sent
            torch.save(value_net.state_dict(), best_model_path)
            print(f"New best model saved with tetris.sent = {best_sent}")

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
    print(device)
    train_agent(num_episodes=1000, gamma=0.95, learning_rate=1e-3, record_interval=10, epsilon_start=1.0, epsilon_end=0.00, epsilon_decay=0.95, device=device)