import numpy as np
import gymnasium as gym
from blackjack import BlackjackEnv 
import matplotlib.pyplot as plt
import math
from qlearningv3 import evaluate_policy, state_to_key, choose_action
import numpy as np
## Q-Learning w PER
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = []
        self.pos = 0

    def add(self, experience, priority):
        """Add a new experience with its priority to the buffer."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
            self.priorities.append(priority)
        else:
            self.buffer[self.pos] = experience
            self.priorities[self.pos] = priority
            self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """Sample a batch of experiences with probability proportional to priority^alpha."""
        scaled_priorities = np.array(self.priorities) ** self.alpha
        sample_probabilities = scaled_priorities / np.sum(scaled_priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=sample_probabilities)
        samples = [self.buffer[i] for i in indices]
        return samples, indices

    def update_priorities(self, indices, new_priorities):
        """Update the priorities of sampled experiences."""
        for idx, priority in zip(indices, new_priorities):
            self.priorities[idx] = priority

# Modified training function with Prioritized Experience Replay
def run_training(initial_epsilon, num_train_episodes, alpha, gamma,
                 replay_capacity=10000, batch_size=32, replay_updates=5):
    
    # Initialize environment (assumes BlackjackEnv is defined elsewhere)
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    n_actions = env.action_space.n
    Q = {}
    training_rewards = []
    epsilon = initial_epsilon

    # Prioritized replay buffer
    replay_buffer = PrioritizedReplayBuffer(capacity=replay_capacity, alpha=0.6)
    
    # Set rolling window and wait threshold to 10% of total training episodes.
    rolling_window = int(0.1 * num_train_episodes)
    wait_threshold = int(0.1 * num_train_episodes)
    best_rolling_avg = -np.inf
    episodes_since_improvement = 0

    print(f"\nStarting training: {num_train_episodes} episodes, initial epsilon={epsilon}")
    for episode in range(num_train_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        episode_reward = 0.0
        done = False

        while not done:
            action = choose_action(state, Q, n_actions, epsilon)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_to_key(next_obs)
            
            # Initialize Q-values for unseen states
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            if next_state not in Q:
                Q[next_state] = np.zeros(n_actions)
            
            # Compute TD error for current transition (before update)
            old_value = Q[state][action]
            next_max = np.max(Q[next_state])
            td_target = reward + gamma * next_max * (not done)
            td_error = abs(td_target - old_value)
            
            # Q-learning update (online update)
            Q[state][action] = old_value + alpha * (td_target - old_value)
            
            # Add experience to replay buffer with current TD error as priority
            
            replay_buffer.add((state, action, reward, next_state, done), td_error)
            
            state = next_state
            episode_reward += reward

        training_rewards.append(episode_reward)
        
        # Adaptive epsilon decay based on rolling average rewards
        if episode >= rolling_window:
            current_rolling_avg = np.mean(training_rewards[-rolling_window:])
            if current_rolling_avg > best_rolling_avg:
                best_rolling_avg = current_rolling_avg
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1
                if episodes_since_improvement >= wait_threshold and epsilon > 0.15:
                    epsilon = max(epsilon * 0.99, 0.15)
                    episodes_since_improvement = 0  # Reset counter after decaying epsilon
        
        # Perform additional replay updates from the prioritized buffer
        if len(replay_buffer.buffer) >= batch_size:
            for _ in range(replay_updates):
                samples, indices = replay_buffer.sample(batch_size)
                new_priorities = []
                for sample in samples:
                    s, a, r, next_s, d = sample
                    if s not in Q:
                        Q[s] = np.zeros(n_actions)
                    if next_s not in Q:
                        Q[next_s] = np.zeros(n_actions)
                    # Compute new TD error for the sampled transition
                    td_target_sample = r + gamma * np.max(Q[next_s]) * (not d)
                    td_error_sample = abs(td_target_sample - Q[s][a])
                    # Update Q-table for the sampled transition
                    Q[s][a] = Q[s][a] + alpha * (td_target_sample - Q[s][a])
                    new_priorities.append(td_error_sample)
                # Update priorities in the buffer for the sampled transitions
                replay_buffer.update_priorities(indices, new_priorities)
        
        # .
        if (episode + 1) % (num_train_episodes // 10) == 0:
            print(f"  Episode {episode + 1}/{num_train_episodes} completed, current epsilon: {epsilon:.4f}")

    env.close()
    return Q, training_rewards

# --- Experiment Setup and Execution ---
import matplotlib.pyplot as plt

# Define hyperparameters.
alpha = 0.1        # Learning rate
gamma = 1.0        # Discount factor
num_eval_episodes = 100000  # Number of evaluation episodes


episodes_list = [1000, 5000, 10000, 50000, 100000, 200000, 500000]
win_rates = []


for episodes in episodes_list:
    print(f"\nTraining with {episodes} episodes")
    Q_temp, _ = run_training(initial_epsilon=1.0, 
                             num_train_episodes=episodes, 
                             alpha=alpha, 
                             gamma=gamma,
                             replay_capacity=10000,  # Maximum transitions in replay buffer
                             batch_size=32,          # Mini-batch size for replay updates
                             replay_updates=5)       # Number of replay updates per episode
                             
    avg_reward, win_rate, _, _, _ = evaluate_policy(Q_temp, num_eval_episodes)
    win_rates.append(win_rate)
    print(f"  Evaluation: Avg. Reward = {avg_reward:.3f}, Win Rate = {win_rate:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(episodes_list, win_rates, marker='o', linestyle='-')
plt.title("Win Rate vs. Training Episodes with Prioritized Experience Replay")
plt.xlabel("Number of Training Episodes")
plt.ylabel("Win Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()