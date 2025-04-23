import torch
import torch.nn as nn
import torch.optim as optim
from TetrisBattle.settings import *
from TetrisBattle.envs.tetris_env import TetrisSingleEnv
from video_recorder import VideoRecorder
import csv
import random


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
        step = 0
        ep_return = 0
        episode_data = []   
        count = 0
        ep_loss = 0

        # get initial action sequence
        # Access the Tetris instance
        tetris = env.game_interface.tetris_list[0]["tetris"]

        # Get all possible states
        final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

        # Compute r(s'|s) + V(s') for each possible state
        future_values = []
        for grid, reward in zip(final_states, rewards):
            grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
            value = value_net(grid_tensor).item()
            future_values.append(reward + value)

        # Select the best future state
        # Epsilon-greedy selection
        if random.random() < epsilon:  # Exploration
            best_index = random.randint(0, len(final_states) - 1)
        else:  # Exploitation
            best_index = future_values.index(max(future_values))
        best_index = future_values.index(max(future_values))
        best_action_sequence = action_sequences[best_index]
        best_future_state = final_states[best_index]
        best_reward = rewards[best_index]

        state, _, _, _ = env.step(0)  # Initialize the environment with a no-op action

        episode_data.append((state["grid"][1], best_reward, best_future_state))
        ep_return += best_reward

        while not done:
            if step >= len(best_action_sequence):
                # Access the Tetris instance
                tetris = env.game_interface.tetris_list[0]["tetris"]

                # Get all possible states
                final_states, action_sequences, was_held, rewards = tetris.get_all_possible_states()

                # Compute r(s'|s) + V(s') for each possible state
                future_values = []
                for grid, reward in zip(final_states, rewards):
                    grid_tensor = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)  # Add batch and channel dims
                    value = value_net(grid_tensor).item()
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

                step = 0
                ep_return += best_reward
                count += 1

                episode_data.append((state["grid"][1], best_reward, best_future_state))

            # Execute the action sequence
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
            total_sent = tetris.sent
            recorder.capture_frame(env.game_interface.screen)

        for s, r, s_prime in episode_data:
            # Convert states to tensors
            s_tensor = torch.tensor(s, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)
            s_prime_tensor = torch.tensor(s_prime, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

            # Compute target y = gamma * [r + V(s')]
            with torch.no_grad():
                v_s_prime = value_net(s_prime_tensor).item()
            target = gamma * (r + v_s_prime)

            # Compute loss
            v_s = value_net(s_tensor)
            loss = nn.MSELoss()(v_s, torch.tensor([[target]], dtype=torch.float32).to(device))
            torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
            ep_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
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
    #print(device)

    # Move the value network to the GPU
    value_net = ValueNetwork().to(device)
    train_agent(num_episodes=1000, gamma=0.95, learning_rate=1e-3, record_interval=10, epsilon_start=1.0, epsilon_end=0.00, epsilon_decay=0.95, device="cpu")