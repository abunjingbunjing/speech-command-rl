class ThresholdTuningAgent:
    """
    State:  current threshold (discretized into bins)
    Action: raise threshold / lower threshold / keep same
    Reward: based on cost of false accepts and false rejects

    A false accept  = accepting a wrong prediction (dangerous, high cost)
    A false reject  = rejecting a correct prediction (annoying, low cost)
    """

    def __init__(self, n_bins=10, lr=0.1, gamma=0.9, epsilon=0.3):
        self.n_bins    = n_bins
        self.lr        = lr       # learning rate
        self.gamma     = gamma    # discount factor
        self.epsilon   = epsilon  # exploration rate (epsilon-greedy)
        self.n_actions = 3        # 0=lower, 1=keep, 2=raise
        # Q-table: rows = threshold states, cols = actions
        self.q_table   = np.zeros((n_bins, self.n_actions))
        self.threshold_values = np.linspace(0.1, 0.99, n_bins)
        self.current_bin = n_bins // 2

    @property
    def threshold(self):
        return self.threshold_values[self.current_bin]

    def get_action(self):
        """Epsilon-greedy: explore randomly or exploit best known action."""
        if random.random() < self.epsilon:
            return random.randint(0, self.n_actions - 1)
        return int(np.argmax(self.q_table[self.current_bin]))

    def apply_action(self, action):
        """Move threshold up, down, or stay."""
        if action == 0:
            self.current_bin = max(0, self.current_bin - 1)
        elif action == 2:
            self.current_bin = min(self.n_bins - 1, self.current_bin + 1)

    def compute_reward(self, probs, true_labels,
                       false_accept_cost=5.0, false_reject_cost=1.0):
        """
        Compute reward for current threshold.
        High confidence = accept prediction.
        Low confidence = reject (say 'uncertain').
        """
        max_probs  = probs.max(axis=1)      # highest class probability
        pred_class = probs.argmax(axis=1)   # predicted class

        accepted = max_probs >= self.threshold
        correct  = (pred_class == true_labels)

        false_accepts  = np.sum( accepted & ~correct)  # accepted but wrong
        false_rejects  = np.sum(~accepted &  correct)  # rejected but right
        true_accepts   = np.sum( accepted &  correct)  # accepted and right

        reward = (true_accepts
                  - false_accept_cost  * false_accepts
                  - false_reject_cost  * false_rejects)
        return float(reward)

    def update(self, state, action, reward, next_state):
        """Q-learning update rule."""
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next
        self.q_table[state, action] += self.lr * (td_target - self.q_table[state, action])

# RL Agent Training
agent = ThresholdTuningAgent(n_bins=10, epsilon=0.3)
N_EPISODES = 200
rewards_per_episode = []

print("Training RL agent...\n")

for episode in range(N_EPISODES):
    # Batch sampling
    idx = np.random.choice(len(all_probs), size=200, replace=False)
    batch_probs  = all_probs[idx]
    batch_labels = all_true[idx]

    state = agent.current_bin
    total_reward = 0

    for _ in range(10):  # 10 steps per episode
        action = agent.get_action()
        agent.apply_action(action)
        reward     = agent.compute_reward(batch_probs, batch_labels)
        next_state = agent.current_bin
        agent.update(state, action, reward, next_state)
        state       = next_state
        total_reward += reward

    # Decay epsilon (explore less over time)
    agent.epsilon = max(0.05, agent.epsilon * 0.995)
    rewards_per_episode.append(total_reward)

    if (episode + 1) % 50 == 0:
        print(f"Episode {episode+1} | Avg reward: {np.mean(rewards_per_episode[-50:]):.2f} "
              f"| Threshold: {agent.threshold:.3f} | Epsilon: {agent.epsilon:.3f}")

print(f"\nFinal learned threshold: {agent.threshold:.3f}")
