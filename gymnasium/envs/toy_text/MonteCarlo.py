import numpy as np
import math
import random
from blackjack import BlackjackEnv
import matplotlib.pyplot as plt


# -------------------------------------------------------------------------
# Utility functions for Blackjack
# -------------------------------------------------------------------------
def cmp(a, b):
    """Compare two values; returns +1 if a > b, -1 if a < b, else 0."""
    return float(a > b) - float(a < b)


def draw_card():
    """Draw a card from an infinite deck (1-13, face cards = 10)."""
    return random.randint(1, 13)


def card_value(card):
    """Return the blackjack value of a card (face cards count as 10)."""
    return 1 if card == 1 else min(card, 10)


def sum_hand(hand):
    """
    Return the best score for a hand.
    Aces (value=1) can count as 11 if it doesn't bust the hand.
    """
    total = sum(card_value(card) for card in hand)
    # If there's at least one Ace (card=1) and total + 10 <=21, add +10
    if 1 in hand and total + 10 <= 21:
        return total + 10
    return total


def is_bust(hand):
    """Return True if the hand is bust (>21)."""
    return sum_hand(hand) > 21


def is_natural(hand):
    """Return True if hand is a natural blackjack (2 cards totaling 21)."""
    return len(hand) == 2 and sum_hand(hand) == 21


def simulate_dealer_play(dealer_hand):
    """Dealer hits until sum >=17. Return final dealer hand."""
    hand = list(dealer_hand)
    while sum_hand(hand) < 17:
        hand.append(draw_card())
    return hand


def outcome(player_hand, dealer_hand, natural_bonus=True):
    """
    Returns the reward value based on game outcome.
    - +1.5 for natural blackjack (if natural_bonus=True)
    - +1 for regular win
    - 0 for push (tie)
    - -1 for loss/bust
    Dealer hits to at least 17.
    """
    # Check for player bust
    if is_bust(player_hand):
        return -1

    # Process natural blackjack
    player_natural = is_natural(player_hand)

    # Complete dealer's hand
    dealer_final = simulate_dealer_play(dealer_hand)
    dealer_natural = is_natural(dealer_final)

    # Check for dealer bust
    if is_bust(dealer_final):
        return 1.5 if (player_natural and natural_bonus) else 1

    # If both have naturals, it's a push
    if player_natural and dealer_natural:
        return 0

    # Player has natural, dealer doesn't
    if player_natural and not dealer_natural and natural_bonus:
        return 1.5

    # Compare final hands
    comparison = cmp(sum_hand(player_hand), sum_hand(dealer_final))
    return comparison  # +1 for win, 0 for tie, -1 for loss


# -------------------------------------------------------------------------
# BASIC STRATEGY ROLLOUT POLICY
# -------------------------------------------------------------------------
def basic_strategy_action(player_hand, dealer_hand):
    """
    0 = stick, 1 = hit.
    - If total < 12: always hit
    - If total between 12 and 16: hit if dealer upcard >= 7, else stick
    - If total >= 17: stick
    """
    total = sum_hand(player_hand)
    dealer_upcard = dealer_hand[0]
    upcard_val = card_value(dealer_upcard)

    if total < 12:
        return 1
    elif 12 <= total < 17:
        return 1 if upcard_val >= 7 else 0
    else:
        return 0


# -------------------------------------------------------------------------
# MCTS Implementation with Basic Strategy Rollout
# -------------------------------------------------------------------------
SIMULATION_DEPTH = 10
MCTS_ITERATIONS = 600
EXPLORATION_CONST = 5.0

# MCTS with Basic Strategy Rollout
Q = {}
N = {}
visited_states = set()


def rollout(state):
    """Rollout using the basic strategy policy."""
    player_hand, dealer_hand = list(state[0]), list(state[1])
    while True:
        if is_bust(player_hand):
            return -1
        action = basic_strategy_action(player_hand, dealer_hand)
        if action == 1:  # hit
            player_hand.append(draw_card())
        else:  # stick
            return outcome(player_hand, dealer_hand)


def simulate(state, depth):
    """One recursive MCTS simulation from the given state."""
    if is_bust(state[0]):
        return -1
    if depth == 0:
        return rollout(state)

    if state not in visited_states:
        for action in [0, 1]:
            N[(state, action)] = 0
            Q[(state, action)] = 0.0
        visited_states.add(state)
        return rollout(state)

    total_N = sum(N.get((state, a), 0) for a in [0, 1])
    best_value = -float("inf")
    best_action = None
    for action in [0, 1]:
        n_sa = N.get((state, action), 0)
        q_sa = Q.get((state, action), 0.0)
        uct_value = q_sa + EXPLORATION_CONST * math.sqrt(
            math.log(total_N + 1) / (n_sa + 1)
        )
        if uct_value > best_value:
            best_value = uct_value
            best_action = action

    player_hand, dealer_hand = list(state[0]), list(state[1])
    if best_action == 0:
        sim_reward = outcome(player_hand, dealer_hand)
        value = sim_reward
    else:
        player_hand.append(draw_card())
        next_state = (tuple(player_hand), tuple(dealer_hand))
        value = simulate(next_state, depth - 1)

    key = (state, best_action)
    N[key] = N.get(key, 0) + 1
    Q[key] = Q.get(key, 0.0) + (value - Q[key]) / N[key]
    return value


def mcts_select_action(current_state):
    """MCTS action selection using basic strategy rollout."""
    global visited_states
    visited_states = set()
    for _ in range(MCTS_ITERATIONS):
        simulate(current_state, SIMULATION_DEPTH)

    # Pick best Q
    best_action, best_q = None, -float("inf")
    for action in [0, 1]:
        q_val = Q.get((current_state, action), 0.0)
        if q_val > best_q:
            best_q = q_val
            best_action = action
    return best_action


def evaluate_mcts_policy(num_episodes=10000):
    """
    Evaluate MCTS with basic strategy rollout.
    """
    wins = busts = draws = losses = 0
    total = 0
    true_rewards = []  # Track actual rewards from games
    expected_values = []  # Continue to track MCTS Q-values for comparison
    confidence_scores = []

    for _ in range(num_episodes):
        env = BlackjackEnv(natural=True)
        obs, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            player_hand = list(env.player_hands[env.current_hand_index])
            dealer_hand = list(env.dealer)
            current_state = (tuple(player_hand), tuple(dealer_hand))

            # Reset the MCTS tree for each decision
            global Q, N
            Q = {}
            N = {}

            action = mcts_select_action(current_state)

            # Confidence = |Q(stick) - Q(hit)|
            q_stick = Q.get((current_state, 0), 0.0)
            q_hit = Q.get((current_state, 1), 0.0)
            confidence_scores.append(abs(q_stick - q_hit))

            # Expected value of chosen action
            if (current_state, action) in Q:
                expected_values.append(Q[(current_state, action)])

            # Take a step in the environment
            obs, reward, done, _, _ = env.step(action)
            episode_reward = reward  # For tracking final reward

        # Store the true reward from the episode
        true_rewards.append(episode_reward)
        total += 1

        # Check the final outcome based on the reward
        if episode_reward > 0:
            wins += 1
        elif episode_reward == 0:
            draws += 1
        else:  # reward < 0
            # Check if the player busted by looking at their final hand sum
            player_hand = list(env.player_hands[env.current_hand_index])
            if is_bust(player_hand):
                busts += 1
            else:
                losses += 1

    # Note: loss_rate includes busts (total negative outcomes)
    return {
        "win_rate": wins / total,
        "bust_rate": busts / total,
        "draw_rate": draws / total,
        "loss_rate": (losses + busts) / total,  # Combined losses and busts
        "pure_loss_rate": losses / total,  # Losses without busts
        "avg_ev": np.mean(expected_values) if expected_values else 0,  # MCTS Q-values
        "avg_confidence": np.mean(confidence_scores) if confidence_scores else 0,
        "true_ev": np.mean(true_rewards) if true_rewards else 0,  # True game rewards
        "all_true_rewards": true_rewards,  # Store all rewards for batch calculations
    }


def evaluate_mcts_progress(num_batches=50, batch_size=200):
    """
    Evaluate MCTS over multiple batches.
    Returns dict of lists for each metric.
    """
    metrics_history = {
        "win_rates": [],
        "bust_rates": [],
        "draw_rates": [],
        "loss_rates": [],
        "avg_evs": [],
        "true_evs": [],
        "avg_confidences": [],
    }

    for i in range(num_batches):
        metrics = evaluate_mcts_policy(num_episodes=batch_size)
        metrics_history["win_rates"].append(metrics["win_rate"])
        metrics_history["bust_rates"].append(metrics["bust_rate"])
        metrics_history["draw_rates"].append(metrics["draw_rate"])
        metrics_history["loss_rates"].append(
            metrics["loss_rate"]
        )  # This now includes busts
        metrics_history["avg_evs"].append(metrics["avg_ev"])
        metrics_history["true_evs"].append(metrics["true_ev"])
        metrics_history["avg_confidences"].append(metrics["avg_confidence"])

        print(
            f"Batch {(i+1)*batch_size} episodes: Win rate = {metrics['win_rate']:.2%}, "
            f"Bust rate = {metrics['bust_rate']:.2%}, EV = {metrics['avg_ev']:.4f}, True EV = {metrics['true_ev']:.4f}"
        )

    return metrics_history


# -------------------------------------------------------------------------
# Main Execution
# -------------------------------------------------------------------------
if __name__ == "__main__":
    print("Evaluating MCTS-based Blackjack agent (with Basic Strategy Rollout)...")

    # Batch evaluation for progress tracking and final metrics
    num_batches = 50
    batch_size = 200
    total_episodes = num_batches * batch_size  # Total episodes will be 10,000

    metrics_history = {
        "win_rates": [],
        "bust_rates": [],
        "draw_rates": [],
        "loss_rates": [],
        "avg_evs": [],  # MCTS Q-values (internal estimates)
        "true_evs": [],  # Actual game rewards (true outcomes)
        "avg_confidences": [],
    }

    # Accumulate overall statistics
    total_wins = 0
    total_busts = 0
    total_draws = 0
    total_losses = 0
    all_expected_values = []
    all_confidence_scores = []
    all_true_rewards = []  # Track all actual rewards

    for i in range(num_batches):
        metrics = evaluate_mcts_policy(num_episodes=batch_size)

        # Track batch results for plotting
        metrics_history["win_rates"].append(metrics["win_rate"])
        metrics_history["bust_rates"].append(metrics["bust_rate"])
        metrics_history["draw_rates"].append(metrics["draw_rate"])
        metrics_history["loss_rates"].append(
            metrics["loss_rate"]
        )  # This now includes busts
        metrics_history["avg_evs"].append(metrics["avg_ev"])
        metrics_history["true_evs"].append(metrics["true_ev"])
        metrics_history["avg_confidences"].append(metrics["avg_confidence"])

        # Track overall stats for final metrics
        total_wins += int(metrics["win_rate"] * batch_size)
        total_busts += int(metrics["bust_rate"] * batch_size)
        total_draws += int(metrics["draw_rate"] * batch_size)
        total_losses += int(
            metrics["pure_loss_rate"] * batch_size
        )  # Track actual losses separately

        # Print batch results
        print(
            f"Batch {(i+1)*batch_size} episodes: Win rate = {metrics['win_rate']:.2%}, "
            f"Bust rate = {metrics['bust_rate']:.2%}, Loss rate = {metrics['pure_loss_rate']:.2%}, EV = {metrics['avg_ev']:.4f}, True EV = {metrics['true_ev']:.4f}"
        )

    # Calculate final overall metrics
    final_win_rate = total_wins / total_episodes
    final_bust_rate = total_busts / total_episodes
    final_draw_rate = total_draws / total_episodes
    final_loss_rate = (total_losses + total_busts) / total_episodes  # Combined losses

    # Calculate average metrics across all batches
    final_avg_ev = np.mean(metrics_history["avg_evs"])
    final_avg_true_ev = np.mean(metrics_history["true_evs"])
    final_avg_confidence = np.mean(metrics_history["avg_confidences"])
    final_avg_win_rate = np.mean(metrics_history["win_rates"])

    # Print final statistics
    print("\nFinal Statistics after", total_episodes, "episodes:")
    print(f"  Win rate (total):   {final_win_rate:.2%}")
    print(f"  Win rate (avg):     {final_avg_win_rate:.2%}")
    print(f"  Bust rate:          {final_bust_rate:.2%}")
    print(f"  Draw rate:          {final_draw_rate:.2%}")
    print(f"  Loss rate (w/busts):{final_loss_rate:.2%}")
    print(f"  Avg EV:             {final_avg_ev:.4f}")
    print(f"  Avg True EV:        {final_avg_true_ev:.4f}")
    print(f"  Avg Confidence:     {final_avg_confidence:.4f}")

    # Create an x-axis for plotting
    episodes = [(i + 1) * batch_size for i in range(num_batches)]

    # Create figure with 2x2 subplots
    plt.figure(figsize=(15, 10))

    # (A) Win Rate
    plt.subplot(2, 2, 1)
    plt.plot(
        episodes, metrics_history["win_rates"], marker="o", linestyle="-", color="blue"
    )
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="Break-even (50%)")
    plt.axhline(
        y=final_win_rate,
        color="green",
        linestyle="-.",
        alpha=0.7,
        label=f"Average Win Rate ({final_win_rate:.2%})",
    )
    plt.xlabel("Number of Episodes")
    plt.ylabel("Win Rate")
    plt.title("MCTS Win Rate Over Batches")
    plt.legend()
    plt.grid(True)

    # (B) Expected Value (MCTS Q-values)
    plt.subplot(2, 2, 2)
    plt.plot(
        episodes, metrics_history["avg_evs"], marker="s", linestyle="-", color="green"
    )
    plt.axhline(
        y=0.0, color="red", linestyle="--", alpha=0.7, label="Break-even (EV=0)"
    )
    plt.axhline(
        y=final_avg_ev,
        color="blue",
        linestyle="-.",
        alpha=0.7,
        label=f"Average EV ({final_avg_ev:.4f})",
    )
    plt.xlabel("Number of Episodes")
    plt.ylabel("Expected Value")
    plt.title("MCTS Expected Value")
    plt.legend()
    plt.grid(True)

    # (C) True Expected Value
    plt.subplot(2, 2, 3)
    plt.plot(
        episodes, metrics_history["true_evs"], marker="^", linestyle="-", color="purple"
    )
    plt.axhline(
        y=0.0, color="red", linestyle="--", alpha=0.7, label="Break-even (True EV=0)"
    )
    plt.axhline(
        y=final_avg_true_ev,
        color="blue",
        linestyle="-.",
        alpha=0.7,
        label=f"Average True EV ({final_avg_true_ev:.4f})",
    )
    plt.xlabel("Number of Episodes")
    plt.ylabel("True Expected Value")
    plt.title("MCTS True Expected Value")
    plt.legend()
    plt.grid(True)

    # (D) Outcome Distribution Breakdown (Wins, Busts, Draws, Losses)
    plt.subplot(2, 2, 4)
    plt.plot(
        episodes,
        metrics_history["win_rates"],
        marker="o",
        linestyle="-",
        color="blue",
        label="Win Rate",
    )
    plt.plot(
        episodes,
        metrics_history["bust_rates"],
        marker="s",
        linestyle="-",
        color="red",
        label="Bust Rate",
    )
    plt.plot(
        episodes,
        metrics_history["draw_rates"],
        marker="^",
        linestyle="-",
        color="green",
        label="Draw Rate",
    )
    plt.plot(
        episodes,
        metrics_history["loss_rates"],
        marker="d",
        linestyle="-",
        color="orange",
        label="Loss Rate",
    )
    plt.xlabel("Number of Episodes")
    plt.ylabel("Rate")
    plt.title("Outcome Distribution Breakdown")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
