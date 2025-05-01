import numpy as np
import gymnasium as gym
from blackjack import BlackjackEnv 
import matplotlib.pyplot as plt
import math
#Adaptive Epsilon Decay Version
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

def run_training(initial_epsilon, num_train_episodes, alpha, gamma):
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    n_actions = env.action_space.n
    Q = {}
    training_rewards = []
    epsilon = initial_epsilon

    # Variables to track improvement in the rolling average reward
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
            
            # Q-learning update
            old_value = Q[state][action]
            next_max = np.max(Q[next_state])
            Q[state][action] = old_value + alpha * (reward + gamma * next_max - old_value)
            
            state = next_state
            episode_reward += reward

        training_rewards.append(episode_reward)
        
        # Only check for improvement after at least 100 episodes
        if episode >= 100:
            current_rolling_avg = np.mean(training_rewards[-100:])
            if current_rolling_avg > best_rolling_avg:
                best_rolling_avg = current_rolling_avg
                episodes_since_improvement = 0
            else:
                episodes_since_improvement += 1
                if episodes_since_improvement > 100 and epsilon > 0.15:
                    epsilon = max(epsilon * 0.99, 0.15)
                    episodes_since_improvement = 0  
        
        if (episode + 1) % (num_train_episodes // 10) == 0:
            print(f"  Episode {episode + 1}/{num_train_episodes} completed, current epsilon: {epsilon:.4f}")

    env.close()
    return Q, training_rewards

def cmp(a, b):
    return float(a > b) - float(a < b)

import matplotlib.pyplot as plt

def evaluate_policy(Q, num_eval_episodes):
    """
    Evaluates the learned policy (greedy with respect to Q) over a number of episodes.
    
    Returns:
        avg_reward - average reward per episode,
        win_rate   - percentage of hands won,
        bust_count, push_count, loss_count - counts for the different outcomes (evaluated per hand).
    """
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)
    n_actions = env.action_space.n
    total_reward = 0.0
    win_count = 0
    bust_count = 0
    push_count = 0
    loss_count = 0
    total_hands = 0
    plot_episodes = []
    plot_rewards = []

    for i in range(num_eval_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        episode_reward = 0.0
        done = False

        # Run episode using greedy actions
        while not done:
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            action = np.argmax(Q[state])
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = state_to_key(next_obs)
            done = terminated or truncated

        total_reward += episode_reward

        # After the episode term, compare dealer v player
        dealer_score = env.score(env.dealer)
        for hand in env.player_hands:
            total_hands += 1
            if env.is_bust(hand):
                bust_count += 1
            else:
                outcome = cmp(env.score(hand), dealer_score)
                if outcome > 0:
                    win_count += 1
                elif outcome == 0:
                    push_count += 1
                elif outcome < 0:
                    loss_count += 1

        # Record reward only for every 1000th evaluation episode
        if (i + 1) % 1000 == 0:
            plot_episodes.append(i + 1)
            plot_rewards.append(episode_reward)

    env.close()
    plt.figure(figsize=(10, 5))
    plt.plot(plot_episodes, plot_rewards, marker='o', linestyle='-')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Reward over Every 1000th Episode')
    plt.grid(True)
    plt.show()

    avg_reward = total_reward / num_eval_episodes
    win_rate = (win_count / total_hands * 100) if total_hands > 0 else 0
    return avg_reward, win_rate, bust_count, push_count, loss_count



alpha = 0.1       
gamma = 1.0     
num_eval_episodes = 150000 

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



