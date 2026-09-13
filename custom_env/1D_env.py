from typing import Optional
import numpy as np
import gymnasium as gym


class environment_1D(gym.Env):

    def __init__(self, size:int = 10):
        self.size = size
        self.llm_location = np.array([-1], dtype=np.int32)
        self.target_location = np.array([-1], dtype=np.int32)
        self.observation_space = gym.spaces.Dict(
                    {"target": gym.spaces.Box(0,size - 1, shape=(1,), dtype=int),
                    "llm": gym.spaces.Box(0,size - 1, shape=(1,), dtype=int),
                    }) 
        
        self.action_space = gym.spaces.Discrete(2)
        self.action_to_direction = {
            0: np.array([-1],dtype=np.int32),
            1 : np.array([1],dtype=np.int32)
        }

    def _get_obs(self):
        return {"llm":self.llm_location, "target":self.target_location}
     
    def _get_info(self):
        return {"distance": abs(self.llm_location[0] - self.target_location[0])}
    
    def reset(self,seed=None,options=None):
           super().reset(seed=seed)
           self.llm_location = self.np_random.integers(0, self.size, size=1, dtype=int) 

           self.target_location = self.llm_location
           while np.array_equal(self.target_location, self.llm_location):
                       self.target_location = self.np_random.integers(0, self.size, size=1, dtype=int) 

           observation = self._get_obs()
           info = self._get_info()

           return observation,info
    def step(self,action):
        direction = self.action_to_direction[action]
        self.llm_location = np.clip(
              self.llm_location+direction,0,self.size-1
        )
        terminated  = np.array_equal(self.llm_location,self.target_location)
        truncated = False
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

gym.register(
    id="gymnasium_env/environment_1D-v0",
    entry_point=environment_1D,
    max_episode_steps=100,
)

env = gym.make("gymnasium_env/environment_1D-v0")
obs,info = env.reset()

episodes = 2
print(obs)
for _ in range(episodes):

    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print(obs, reward, terminated)
    if terminated or truncated:
        obs, info = env.reset()
        