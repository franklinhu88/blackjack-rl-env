import numpy as np
import math
import random
from blackjack import BlackjackEnv
import matplotlib.pyplot as plt

# --- Utility functions for card evaluation (similar to blackjack.py) ---

def cmp(a, b):
    """Compare two values; returns +1 if a > b, -1 if a < b, else 0."""
    return float(a > b) - float(a < b)

def draw_card():
    """
    Draw a card from an infinite deck.
    Cards are represented as integers 1-13.
    For scoring, face cards (>=10) are treated as 10.
    """
    card = random.randint(1, 13)
    return card

def card_value(card):
    """Return the blackjack value of a card (face cards count as 10)."""
    if card == 1:
        return 1  # Ace is handled specially in sum_hand
    return min(card, 10)

def sum_hand(hand):
    """
    Return the best score for a hand.
    Aces can count as 11 if they don't bust the hand.
    """
    total = sum(card_value(card) for card in hand)
    # Count aces: a card of value 1 may be counted as 11 if it doesn't bust.
    if 1 in hand and total + 10 <= 21:
        return total + 10
    return total

def is_bust(hand):
    """Return True if the hand is bust (>21)."""
    return sum_hand(hand) > 21

def simulate_dealer_play(dealer_hand):
    """
    Simulate the dealer's play: hit until the sum is at least 17.
    Returns the final dealer hand.
    """
    hand = list(dealer_hand)
    while sum_hand(hand) < 17:
        hand.append(draw_card())
    return hand

def outcome(player_hand, dealer_hand):
    """
    Given a player's hand and dealer's hand (after dealer play),
    return the reward according to blackjack rules.
    Reward: +1 for win, -1 for loss, 0 for draw.
    """
    if is_bust(player_hand):
        return -1
    dealer_final = simulate_dealer_play(dealer_hand)
    if is_bust(dealer_final):
        return 1
    # Compare totals
    return cmp(sum_hand(player_hand), sum_hand(dealer_final))


# --- MCTS Implementation ---
#
# We restrict the available actions to two:
#   0: Stick (stand) – end your turn and let the dealer play.
#   1: Hit – take one additional card.
#
# We represent a simulation state as a tuple:
#   (player_hand, dealer_hand)
# where each hand is represented as a tuple of card values.
#
# Global dictionaries (the “tree”) store:
#   Q[state, action]: estimated value of taking action in state.
#   N[state, action]: number of times (state, action) has been visited.
#
# The simulation uses a discount factor gamma and exploration constant c.
# (In blackjack, gamma can be set to 1 since rewards are given at the end.)
#

# Global MCTS trees
Q = {}   # key: (state, action) where state = (player_hand, dealer_hand)
N = {}   # key: (state, action)

# Hyperparameters for MCTS simulation
SIMULATION_DEPTH = 10     # maximum simulation depth
MCTS_ITERATIONS = 600     # number of simulations per decision
EXPLORATION_CONST = 5.0   # exploration constant c
GAMMA = 1.0               # discount factor (no discounting in episodic blackjack)

def state_key(player_hand, dealer_hand):
    """
    Create a key for the simulation state.
    We use tuple representations of the hands.
    """
    return (tuple(player_hand), tuple(dealer_hand))

def is_terminal(state):
    """A state is terminal if the player's hand is bust."""
    player_hand, _ = state
    return is_bust(player_hand)

def rollout(state):
    """
    Rollout (default policy) from the given state.
    We use a simple rule: hit if player's total < 17, else stick.
    When sticking, we simulate dealer play and return the outcome.
    """
    player_hand, dealer_hand = list(state[0]), list(state[1])
    # Continue until terminal decision (hit or stick)
    while True:
        if is_bust(player_hand):
            return -1  # bust
        total = sum_hand(player_hand)
        if total < 17:
            # Hit: take one card and continue
            player_hand.append(draw_card())
        else:
            # Stick: simulate dealer play and return outcome
            return outcome(player_hand, dealer_hand)

def simulate(state, depth):
    """
    Recursive MCTS simulation from state with remaining simulation depth.
    Returns the simulated value.
    """
    if is_bust(state[0]):
        # Terminal: player already busted.
        return -1
    if depth == 0:
        return rollout(state)
    
    # If state is not in our tree (i.e. unvisited), initialize and do a rollout.
    if state not in visited_states:
        # Initialize counts for available actions (hit=1, stick=0)
        for action in [0, 1]:
            N[(state, action)] = 0
            Q[(state, action)] = 0.0
        visited_states.add(state)
        return rollout(state)
    
    # Total visits for state (used for UCT)
    total_N = sum(N.get((state, a), 0) for a in [0, 1])
    
    # Select action by UCT
    best_value = -float('inf')
    best_action = None
    for action in [0, 1]:
        n_sa = N.get((state, action), 0)
        q_sa = Q.get((state, action), 0.0)
        # Use UCT formula; add bonus if action not taken before.
        uct_value = q_sa + EXPLORATION_CONST * math.sqrt(math.log(total_N + 1) / (n_sa + 1))
        if uct_value > best_value:
            best_value = uct_value
            best_action = action

    # Simulate next state based on chosen action.
    player_hand, dealer_hand = list(state[0]), list(state[1])
    if best_action == 0:
        # Stick: simulate dealer play immediately and return outcome.
        sim_reward = outcome(player_hand, dealer_hand)
        next_state = None  # Terminal action.
        value = sim_reward
    else:  # best_action == 1, Hit
        # Hit: draw a card and continue the game.
        new_card = draw_card()
        player_hand.append(new_card)
        next_state = state_key(player_hand, dealer_hand)
        value = simulate(next_state, depth - 1)
    
    # Update counts and Q value for (state, best_action)
    key = (state, best_action)
    N[key] = N.get(key, 0) + 1
    Q[key] = Q.get(key, 0.0) + (value - Q.get(key, 0.0)) / N[key]
    
    return value

def mcts_select_action(current_state):
    """
    Run MCTS simulations from current_state for a fixed number of iterations.
    Return the action (0 for stick, 1 for hit) with the highest estimated value.
    """
    global visited_states
    visited_states = set()
    for _ in range(MCTS_ITERATIONS):
        simulate(current_state, SIMULATION_DEPTH)
    # After simulations, choose the action with highest average Q.
    best_action = None
    best_q = -float('inf')
    for action in [0, 1]:
        q_val = Q.get((current_state, action), 0.0)
        if q_val > best_q:
            best_q = q_val
            best_action = action
    return best_action

# --- Evaluation using MCTS for decision making ---
#
# For evaluation we run episodes in the environment.
# At each decision point, we use MCTS to select an action.
# Note: Our simulation restricts actions to hit (1) and stick (0).
# We map the MCTS decision to the environment by:
#   - Using 1 for hit.
#   - Using 0 for stick.
#
def evaluate_mcts_policy(num_episodes=10000):
    wins = 0
    total = 0
    for _ in range(num_episodes):
        env = BlackjackEnv()
        obs, _ = env.reset()
        done = False
        # We use the full internal state from the environment:
        #   player_hand = env.player_hands[0]
        #   dealer_hand = env.dealer
        # (We ignore double and split; if chosen by env randomly these actions are not used.)
        while not done:
            # Extract complete state from the environment
            # (Make a copy so simulation does not affect the env.)
            player_hand = list(env.player_hands[env.current_hand_index])
            dealer_hand = list(env.dealer)
            current_state = state_key(player_hand, dealer_hand)
            # Reset the MCTS tree (global dictionaries) for each decision.
            global Q, N
            Q = {}
            N = {}
            # Run MCTS to select action (0: Stick, 1: Hit)
            action = mcts_select_action(current_state)
            # In our evaluation, we only use hit and stick.
            # (If the environment supports double/split, we treat them as stick.)
            if action is None:
                action = 0
            obs, reward, done, _, _ = env.step(action)
        # A win is defined as a positive final reward.
        if reward > 0:
            wins += 1
        total += 1
    win_rate = wins / total
    return win_rate

def evaluate_mcts_progress(num_batches=50, batch_size=200):
    """
    Evaluate the MCTS policy in batches and record win rates.
    
    Parameters:
    - num_batches: Number of batches to evaluate.
    - batch_size: Number of episodes per batch.
    
    Returns:
    - A list of win rates for each batch.
    """
    batch_win_rates = []
    for i in range(num_batches):
        win_rate = evaluate_mcts_policy(num_episodes=batch_size)
        batch_win_rates.append(win_rate)
        print(f"Batch {(i+1)*batch_size} episodes: Win rate = {win_rate:.2%}")
    return batch_win_rates


if __name__ == '__main__':
    print("Evaluating MCTS-based Blackjack agent...")
    final_win_rate = evaluate_mcts_policy(num_episodes=5000)
    print(f"Final win rate: {final_win_rate:.2%}")

    # Evaluate performance progress over batches
    win_rates = evaluate_mcts_progress(num_batches=50, batch_size=200)
    
    # Plot the win rate progress
    # episodes = [ (i+1)*200 for i in range(len(win_rates)) ]
    # plt.figure(figsize=(10, 5))
    # plt.plot(episodes, win_rates, marker='o')
    # plt.xlabel("Episodes Evaluated")
    # plt.ylabel("Win Rate")
    # plt.title("Win Rate Progress of MCTS-based Blackjack Agent")
    # plt.grid(True)
    # plt.show()
