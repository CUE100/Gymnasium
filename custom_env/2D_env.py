from typing import Optional
import numpy as np
import gymnasium as gym


class environment(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, size: int = 5, render_mode: Optional[str] = None):
        self.size = size
        self.render_mode = render_mode 

        self._llm_location = np.array([-1, -1], dtype=np.int32)
        self._target_location = np.array([-1, -1], dtype=np.int32)

        self.observation_space = gym.spaces.Dict(
            {
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "llm": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
            }
        )

        self.action_space = gym.spaces.Discrete(4)
        self._action_to_direction = {
            0: np.array([0, 1]),
            1: np.array([-1, 0]),
            2: np.array([0, -1]),
            3: np.array([1, 0]),
        }

    def _get_obs(self):
        return {"llm": self._llm_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._llm_location - self._target_location, ord=1
            )
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        self._llm_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        self._target_location = self._llm_location
        while np.array_equal(self._target_location, self._llm_location):
            self._target_location = self.np_random.integers(0, self.size, size=2, dtype=int)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        direction = self._action_to_direction[action]
        self._llm_location = np.clip(self._llm_location + direction, 0, self.size - 1)

        terminated = np.array_equal(self._llm_location, self._target_location)
        truncated = False
        reward = 1 if terminated else 0

        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            for y in range(self.size - 1, -1, -1):
                row = ""
                for x in range(self.size):
                    if np.array_equal([x, y], self._llm_location):
                        row += "A "
                    elif np.array_equal([x, y], self._target_location):
                        row += "T "
                    else:
                        row += ". "
                print(row)
            print()


gym.register(
    id="gymnasium_env/environment-v0",
    entry_point=environment,
    max_episode_steps=300,
)

env = gym.make("gymnasium_env/environment-v0", size=5, render_mode="human")
obs, info = env.reset()
print(obs)
env.render()

for _ in range(5):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    print(obs, reward, terminated)
    if terminated or truncated:
        obs, info = env.reset()

    