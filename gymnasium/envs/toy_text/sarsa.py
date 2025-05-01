import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from blackjack import BlackjackEnv


# Convert environment state to a hashable key (tuple form)
def state_to_key(state):
    if isinstance(state, np.ndarray):
        return tuple(state.tolist())
    elif isinstance(state, (list, tuple)):
        return tuple(state_to_key(s) for s in state)
    else:
        return state


# Compute softmax probabilities for action selection
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


# Choose an action using softmax exploration strategy
def choose_action_softmax(state, Q, n_actions, temperature):
    if state not in Q:
        Q[state] = np.zeros(n_actions)
    probs = softmax(Q[state] / temperature)
    return np.random.choice(n_actions, p=probs)


# Evaluate the learned policy over multiple episodes
def evaluate_policy(env, Q, n_actions, num_eval_episodes=1000):
    win_count = 0
    total_reward = 0.0
    for _ in range(num_eval_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        done = False
        episode_reward = 0.0
        while not done:
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            action = np.argmax(Q[state])  # Always take the greedy action during evaluation
            next_obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            state = state_to_key(next_obs)
            done = terminated or truncated
        total_reward += episode_reward
        if episode_reward > 0.0:
            win_count += 1
    avg_reward = total_reward / num_eval_episodes
    win_rate = win_count / num_eval_episodes * 100
    return avg_reward, win_rate


def main():
    # Initialize environment
    env = BlackjackEnv(render_mode=None, natural=False, sab=False)

    # Hyperparameters
    num_episodes = 1000000
    alpha = 0.05
    gamma = 0.99
    temperature = 1.0
    temperature_min = 0.1
    temperature_decay = 0.99995
    n_actions = env.action_space.n
    Q = {}

    # Lists to store evaluation results
    evaluation_points = []
    win_rates = []
    avg_rewards = []
    eval_interval = 20000  # Evaluate every 20,000 episodes

    print("Starting SARSA training with softmax action selection...")

    # SARSA Training Loop
    for episode in range(num_episodes):
        obs, info = env.reset()
        state = state_to_key(obs)
        action = choose_action_softmax(state, Q, n_actions, temperature)
        done = False

        while not done:
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = state_to_key(next_obs)
            next_action = choose_action_softmax(next_state, Q, n_actions, temperature) if not done else None
            if state not in Q:
                Q[state] = np.zeros(n_actions)
            if not done:
                target = reward + gamma * Q[next_state][next_action]
            else:
                target = reward
            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
            action = next_action

        # Decay temperature to reduce exploration over time
        temperature = max(temperature_min, temperature * temperature_decay)

        # Periodic policy evaluation
        if (episode + 1) % eval_interval == 0:
            avg_reward, win_rate = evaluate_policy(env, Q, n_actions, num_eval_episodes=1000)
            evaluation_points.append(episode + 1)
            avg_rewards.append(avg_reward)
            win_rates.append(win_rate)
            print(f"Episode {episode + 1}: Average Reward = {avg_reward:.3f}, Win Rate = {win_rate:.2f}%")

    print("Training complete.\n")

    # Final Evaluation
    num_eval_episodes = 10000
    total_reward = 0.0
    win_count = 0
    bust_count = 0
    push_count = 0
    loss_count = 0
    eval_rewards = []

    print("Starting final evaluation of the learned policy...")
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
        eval_rewards.append(episode_reward)
        if env.is_bust(env.player_hands[env.current_hand_index]):
            bust_count += 1
        elif episode_reward > 0.0:
            win_count += 1
        elif episode_reward == 0.0:
            push_count += 1
        elif episode_reward < 0.0:
            loss_count += 1

    # Print Final Results
    avg_reward_final = total_reward / num_eval_episodes
    print(f"Final Evaluation Average Reward: {avg_reward_final:.3f}")
    print("Outcomes:")
    print(f"  Win Rate   : {win_count / num_eval_episodes * 100:.3f}%")
    print(f"  Bust Rate  : {bust_count / num_eval_episodes * 100:.3f}%")
    print(f"  Loss Rate  : {loss_count / num_eval_episodes * 100:.3f}%")
    print(f"  Push Rate  : {push_count / num_eval_episodes * 100:.3f}%")

    # ------------------- Plotting the Results -------------------
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(evaluation_points, win_rates, marker='o')
    plt.title("Win Rate Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Win Rate (%)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(evaluation_points, avg_rewards, marker='o', color='orange')
    plt.title("Average Reward Over Training")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    env.close()


if __name__ == "__main__":
    main()