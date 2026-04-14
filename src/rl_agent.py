import numpy as np
import matplotlib.pyplot as plt
import random
import json
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent))
from data_pipeline import COMMANDS, RESULTS_DIR, SEED

random.seed(SEED)
np.random.seed(SEED)


class ThresholdTuningAgent:
    def __init__(self, n_bins=10, lr=0.1, gamma=0.9, epsilon=0.3):
        self.n_bins    = n_bins
        self.lr        = lr
        self.gamma     = gamma
        self.epsilon   = epsilon
        self.n_actions = 3
        self.q_table   = np.zeros((n_bins, self.n_actions))
        self.threshold_values = np.linspace(0.1, 0.99, n_bins)
        self.current_bin = n_bins // 2

    @property
    def threshold(self):
        return self.threshold_values[self.current_bin]

    def get_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[self.current_bin]))

    def apply_action(self, action):
        if action == 0:
            self.current_bin = max(0, self.current_bin - 1)
        elif action == 2:
            self.current_bin = min(self.n_bins - 1, self.current_bin + 1)

    def compute_reward(self, probs, true_labels,
                       false_accept_cost=5.0, false_reject_cost=1.0):
        max_probs  = probs.max(axis=1)
        pred_class = probs.argmax(axis=1)
        accepted   = max_probs >= self.threshold
        correct    = (pred_class == true_labels)

        false_accepts = np.sum( accepted & ~correct)
        false_rejects = np.sum(~accepted &  correct)
        true_accepts  = np.sum( accepted &  correct)

        return float(true_accepts
                     - false_accept_cost * false_accepts
                     - false_reject_cost * false_rejects)

    def update(self, state, action, reward, next_state):
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])


def train_agent(all_probs, all_true, n_episodes=200):
    agent = ThresholdTuningAgent(n_bins=10, epsilon=0.3)
    rewards_per_episode = []

    for episode in range(n_episodes):
        idx = np.random.choice(len(all_probs), size=200, replace=False)
        batch_probs  = all_probs[idx]
        batch_labels = all_true[idx]

        state = agent.current_bin
        total_reward = 0

        for _ in range(10):
            action     = agent.get_action()
            agent.apply_action(action)
            reward     = agent.compute_reward(batch_probs, batch_labels)
            next_state = agent.current_bin
            agent.update(state, action, reward, next_state)
            state        = next_state
            total_reward += reward

        agent.epsilon = max(0.05, agent.epsilon * 0.995)
        rewards_per_episode.append(total_reward)

        if (episode + 1) % 50 == 0:
            print(f"  Episode {episode+1:3d} | "
                  f"Avg reward: {np.mean(rewards_per_episode[-50:]):7.2f} | "
                  f"Threshold: {agent.threshold:.3f} | "
                  f"Epsilon: {agent.epsilon:.3f}")

    return agent, rewards_per_episode


def save_rl_learning_curve(rewards, save_path):
    window = 20
    smoothed = np.convolve(rewards, np.ones(window) / window, mode='valid')
    plt.figure(figsize=(10, 4))
    plt.plot(rewards,  alpha=0.3, color='steelblue', label='Raw reward')
    plt.plot(smoothed, color='steelblue', linewidth=2,
             label=f'{window}-episode moving avg')
    plt.title('RL agent learning curve')
    plt.xlabel('Episode')
    plt.ylabel('Total reward')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"RL learning curve saved to {save_path}")


if __name__ == "__main__":
    # --- Load test probabilities saved by eval.py ---
    probs_path  = RESULTS_DIR / "test_probs.npy"
    labels_path = RESULTS_DIR / "test_labels.npy"

    if not probs_path.exists():
        raise FileNotFoundError(
            f"No test_probs.npy found at {probs_path}. Run eval.py first."
        )

    print("Loading test probabilities from eval.py output...")
    all_probs = np.load(probs_path)
    all_true  = np.load(labels_path)
    print(f"  Loaded {len(all_probs)} samples, {all_probs.shape[1]} classes")

    # --- Train RL agent ---
    print("\nTraining RL threshold-tuning agent...")
    agent, rewards = train_agent(all_probs, all_true, n_episodes=200)

    # --- Save learning curve plot ---
    save_rl_learning_curve(rewards, RESULTS_DIR / "rl_learning_curve.png")

    # --- Evaluate final threshold ---
    final_reward = agent.compute_reward(all_probs, all_true)
    print(f"\nFinal learned threshold : {agent.threshold:.3f}")
    print(f"Final reward on full set: {final_reward:.2f}")

    # --- Compare vs default threshold of 0.5 ---
    agent_default = ThresholdTuningAgent(n_bins=10)
    agent_default.current_bin = 4  # ~0.5 threshold
    default_reward = agent_default.compute_reward(all_probs, all_true)
    print(f"Default threshold (0.5) reward: {default_reward:.2f}")
    print(f"Improvement: {final_reward - default_reward:+.2f}")

    # --- Save RL results as JSON ---
    rl_results = {
        "final_threshold"    : float(agent.threshold),
        "final_reward"       : float(final_reward),
        "default_reward"     : float(default_reward),
        "improvement"        : float(final_reward - default_reward),
        "n_episodes"         : 200,
        "q_table"            : agent.q_table.tolist(),
    }
    with open(RESULTS_DIR / "rl_results.json", "w") as f:
        json.dump(rl_results, f, indent=2)

    print(f"\nRL results saved to {RESULTS_DIR}/rl_results.json")
    print("rl_agent.py complete.")
