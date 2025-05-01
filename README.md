# Blackjack RL Environment and Algorithms for CS4359

This repository is a customized reinforcement learning environment for Blackjack, built on top of the [Gymnasium](https://github.com/Farama-Foundation/Gymnasium) library by the Farama Foundation. Our modifications include the addition of *split* and *double down* actions, expanding the strategic possibilities available to learning agents.

## Features
- Gymnasium-based Blackjack environment
- Supports *hit*, *stand*, *split*, and *double down* actions
- Compatible with various RL algorithms (SARSA, PPO, Monte-Carlo, Q-Learning)

## Getting Started
Ensure you have Python installed and set up a virtual environment. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Running Models
Below are placeholders for running specific models. Fill in details as you develop and test each approach.

### SARSA
```bash
cd gymnasium
cd envs
cd toy_text
python3 sarsa.py
```

### PPO
```bash
# Example command to run PPO
```

### Monte-Carlo
```bash
cd gymnasium
cd envs
cd toy_text
python3 MonteCarlo.py
```

### Q-Learning
```bash
cd gymnasium
cd envs
cd toy_text
python3 qlearningv1.py
python3 qlearningv2.py
python3 qlearningv3.py
python3 qlearningv4.py
python3 qlearningv5.py

```

## Group Members
- Franklin Hu
- Jimmy Baek
- Hunter Qin
- Emre Bilge
- Sona Javadi
- Daniel Zhang

## Acknowledgments
This project is based on the Gymnasium library developed and maintained by the [Farama Foundation](https://github.com/Farama-Foundation). We extend our gratitude for their open-source contributions to the RL community.

