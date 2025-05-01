# ### LEARING RATE SCHEUDLER TRAINING APPROACH
import numpy as np
import gymnasium as gym
from blackjack import BlackjackEnv 
import matplotlib.pyplot as plt
import math
from qlearningv3 import evaluate_policy, state_to_key, choose_action
def run_training(initial_epsilon, num_train_episodes, alpha, gamma):
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    n_actions = env.action_space.n
    Q = {}
    training_rewards = []
    epsilon = initial_epsilon
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
            
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            if next_state not in Q:
                Q[next_state] = np.zeros(n_actions)
            
            old_value = Q[state][action]
            next_max = np.max(Q[next_state])
            Q[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)
            state = next_state
            episode_reward += reward
        training_rewards.append(episode_reward)
        if episode >= rolling_window:
            current_rolling_avg = np.mean(training_rewards[-rolling_window:])
            if current_rolling_avg > best_rolling_avg:
                best_rolling_avg = current_rolling_avg
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1
                if episodes_since_improvement >= wait_threshold and epsilon > 0.15:
                    epsilon = max(epsilon * 0.99, 0.15)
                    episodes_since_improvement = 0  
        
        if (episode + 1) % (num_train_episodes // 10) == 0:
            print(f"  Episode {episode + 1}/{num_train_episodes} completed, current epsilon: {epsilon:.4f}")

    env.close()
    return Q, training_rewards
alpha = 0.1      
gamma = 1.0        
num_eval_episodes = 100000  

episodes_list = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000]
win_rates = []

for episodes in episodes_list:
    print(f"\nTraining with {episodes} episodes")
    Q_temp, _ = run_training(initial_epsilon=1.0, num_train_episodes=episodes, 
                             alpha=alpha, gamma=gamma)
    avg_reward, win_rate, _, _, _ = evaluate_policy(Q_temp, num_eval_episodes)
    win_rates.append(win_rate)
    print(f"  Evaluation: Avg. Reward = {avg_reward:.3f}, Win Rate = {win_rate:.2f}%")
plt.figure(figsize=(8, 5))
plt.plot(episodes_list, win_rates, marker='o', linestyle='-')
plt.title("Win Rate vs. Training Episodes (ε decays to 0.15)")
plt.xlabel("Number of Training Episodes")
plt.ylabel("Win Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
