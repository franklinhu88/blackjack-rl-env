import numpy as np
import gymnasium as gym
from blackjack import BlackjackEnv 
import matplotlib.pyplot as plt
import math

def state_to_key(state):
    if isinstance(state, np.ndarray):
        return tuple(state.tolist())
    elif isinstance(state, (list, tuple)):
        return tuple(state_to_key(s) for s in state)
    else:
        return state

def choose_action(state, Q, n_actions, epsilon):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)
    else:
        return np.argmax(Q[state])

def run_training(initial_epsilon, num_train_episodes, alpha, gamma, epsilon_decay=0.995, epsilon_min=0.01):
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    n_actions = env.action_space.n
    Q = {}
    training_rewards = []
    epsilon = initial_epsilon
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
        
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        if (episode + 1) % (num_train_episodes // 10) == 0:
            print(f"  Episode {episode + 1}/{num_train_episodes} completed, current epsilon: {epsilon:.4f}")

    env.close()
    return Q, training_rewards

def evaluate_policy(Q, num_eval_episodes):
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    n_actions = env.action_space.n
    total_reward = 0.0
    win_count = 0
    bust_count = 0
    push_count = 0
    loss_count = 0
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        done = False
        episode_reward = 0.0
        while not done:
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            action = np.argmax(Q[state])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = state_to_key(next_obs)
            done = terminated or truncated
        total_reward += episode_reward
        if env.is_bust(env.current_hand):
            bust_count += 1
        elif episode_reward > 0.0:
            win_count += 1
        elif episode_reward == 0.0:
            push_count += 1
        elif episode_reward <= 0.0:
            loss_count += 1

    env.close()
    avg_reward = total_reward / num_eval_episodes
    win_rate = win_count / num_eval_episodes * 100
    return avg_reward, win_rate, bust_count, push_count, loss_count
alpha = 0.1      
gamma = 1      
num_eval_episodes = 10000  

episodes_list = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000, 1500000, 2000000]
win_rates = []

print("\nExperiment 3: change epsilon!")
for episodes in episodes_list:
    print(f"\nTraining with {episodes} episodes")
    # Use decaying epsilon: initial_epsilon = 0.9, epsilon_decay and epsilon_min
    Q_temp, _ = run_training(initial_epsilon=1, num_train_episodes=episodes, 
                             alpha=alpha, gamma=gamma, epsilon_decay=0.99, epsilon_min=0.1)
    avg_reward, win_rate, _, _, _ = evaluate_policy(Q_temp, num_eval_episodes)
    win_rates.append(win_rate)
    print(f"  Evaluation: Avg. Reward = {avg_reward:.3f}, Win Rate = {win_rate:.2f}%")

plt.figure(figsize=(8, 5))
plt.plot(episodes_list, win_rates, marker='o', linestyle='-')
plt.title("Win Rate vs. Training Episodes (ε = 0.9 with decay)")
plt.xlabel("Number of Training Episodes")
plt.ylabel("Win Rate (%)")
plt.grid(True)
plt.tight_layout()
plt.show()
alpha = 0.1        
gamma = 1.0        
num_eval_episodes = 100000  

episodes_list = [1000, 5000, 10000, 50000, 100000, 200000, 500000, 1000000, 1500000, 2000000]
win_rates = []

print("\nExperiment 3: change epsilon!")
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
