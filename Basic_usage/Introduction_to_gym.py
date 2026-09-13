import gymnasium as gym

# Create a simple environment perfect for beginners
env = gym.make('CartPole-v1')

# The CartPole environment: balance a pole on a moving cart
# - Simple but not trivial
# - Fast training
# - Clear success/failure criteria

#In reinforcement learning, the classic “agent-environment loop” pictured below represents how learning happens in RL. It’s simpler than it might first appear:

# Agent observes the current situation (like looking at a game screen)
# Agent chooses an action based on what it sees (like pressing a button)
# Environment responds with a new situation and a reward (game state changes, score updates)
# Repeat until the episode ends
