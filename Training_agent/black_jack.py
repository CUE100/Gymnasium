"""Train and evaluate a Q-learning agent for Blackjack."""

from collections import defaultdict

import gymnasium as gym
import numpy as np
from tqdm import tqdm


class BlackjackAgent:
    """Q-learning agent that learns a Blackjack policy."""

    def __init__(
        self,
        blackjack_env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = blackjack_env
        self.q_values = defaultdict(lambda: np.zeros(blackjack_env.action_space.n))
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []

    def get_action(self, state: tuple[int, int, bool]) -> int:
        """Choose an action using an epsilon-greedy policy."""
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))

    def update(
        self,
        state: tuple[int, int, bool],
        selected_action: int,
        received_reward: float,
        episode_terminated: bool,
        next_state: tuple[int, int, bool],
    ):
        """Update the action value using a temporal-difference target."""
        future_q_value = (not episode_terminated) * np.max(self.q_values[next_state])
        target = received_reward + self.discount_factor * future_q_value
        temporal_difference = target - self.q_values[state][selected_action]
        self.q_values[state][selected_action] = (
            self.q_values[state][selected_action] + self.lr * temporal_difference
        )
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration while keeping it above the configured minimum."""
        self.epsilon = max(self.final_epsilon,
                           self.epsilon - self.epsilon_decay)


# Training hyperparameters
LEARNING_RATE = 0.01        # How fast to learn (higher = faster but less stable)
N_EPISODES = 100_000        # Number of hands to practice
START_EPSILON = 1.0         # Start with 100% random actions
EPSILON_DECAY = START_EPSILON / (N_EPISODES / 2)  # Reduce exploration over time
FINAL_EPSILON = 0.1         # Always keep some exploration

# Create environment and agent
env = gym.make("Blackjack-v1", sab=False)
env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=N_EPISODES)

agent = BlackjackAgent(
    blackjack_env=env,
    learning_rate=LEARNING_RATE,
    initial_epsilon=START_EPSILON,
    epsilon_decay=EPSILON_DECAY,
    final_epsilon=FINAL_EPSILON,
)

for episode in tqdm(range(N_EPISODES)):
    # Start a new hand
    current_obs, info = env.reset()
    hand_done = False

    # Play one complete hand
    while not hand_done:
        # Agent chooses action (initially random, gradually more intelligent)
        selected_action = agent.get_action(current_obs)

        # Take action and observe result
        next_obs, reward, terminated, truncated, info = env.step(selected_action)

        # Learn from this experience
        agent.update(current_obs, selected_action, reward, terminated, next_obs)

        # Move to next state
        hand_done = terminated or truncated
        current_obs = next_obs

    # Reduce exploration rate (agent becomes less random over time)
    agent.decay_epsilon()




# Test the trained agent
def test_agent(trained_agent, blackjack_env, num_episodes=1000):
    """Test agent performance without learning or exploration."""
    total_rewards = []

    # Temporarily disable exploration for testing
    old_epsilon = trained_agent.epsilon
    trained_agent.epsilon = 0.0  # Pure exploitation

    for _ in range(num_episodes):
        current_obs, _ = blackjack_env.reset()
        episode_reward = 0
        hand_done = False

        while not hand_done:
            selected_action = trained_agent.get_action(current_obs)
            current_obs, reward, terminated, truncated, _ = blackjack_env.step(
                selected_action
            )
            episode_reward += reward
            hand_done = terminated or truncated

        total_rewards.append(episode_reward)

    # Restore original epsilon
    trained_agent.epsilon = old_epsilon

    win_rate = np.mean(np.array(total_rewards) > 0)
    average_reward = np.mean(total_rewards)

    print(f"Test Results over {num_episodes} episodes:")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Average Reward: {average_reward:.3f}")
    print(f"Standard Deviation: {np.std(total_rewards):.3f}")

# Test your agent
test_agent(agent, env)