import numpy as np
import gymnasium as gym

class Easy21(gym.Env):
    metadata = {"render_modes": ["ansi"]}

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        # observe the player's sum and the dealer's card
        self.observation_space = gym.spaces.MultiDiscrete(np.array([23, 11]))
        self.dealer_sum = None

    def _get_obs(self):
      # clamp the som to be in [0, 22] to avoid negative values and values > 21
      sum = np.clip(self.player_sum, 0, 22)
      return np.array([sum, self.dealer_card])
    
    def _get_info(self):
      return {"dealer_sum": self.dealer_sum}
    
    def reset(self, seed=None):
        super().reset(seed=seed)

        self.player_sum = self.np_random.integers(1, 10, endpoint=True)
        self.dealer_card = self.np_random.integers(1, 10, endpoint=True)
        self.dealer_sum = self.dealer_card
        return self._get_obs(), self._get_info()
    
    def _update_dealer_sum(self):
        card = self.np_random.integers(1, 10, endpoint=True)
        color = self.np_random.choice([-1, 1], p=[1 / 3, 2 / 3])
        self.dealer_sum += (card * color)
    
    def step(self, action):
        assert self.action_space.contains(action)
        if action == 0: # hit
            card = self.np_random.integers(1, 10, endpoint=True)
            color = self.np_random.choice([-1, 1], p=[1 / 3, 2 / 3])
            self.player_sum += (card * color)
            if self.player_sum > 21 or self.player_sum < 1:
                return self._get_obs(), -1, True, False, self._get_info()
            else:
                self._update_dealer_sum()
                if self.dealer_sum > 21 or self.dealer_sum < 1:
                    return self._get_obs(), 1, True, False, self._get_info()
                return self._get_obs(), 0, False, False, self._get_info()
        else: # stick
            while 0 < self.dealer_sum < 17: # play out the round
                self._update_dealer_sum()
            if self.dealer_sum > 21 or self.dealer_sum < 1:
                return self._get_obs(), 1, True, False, self._get_info()
            elif self.dealer_sum > self.player_sum:
                return self._get_obs(), -1, True, False, self._get_info()
            elif self.dealer_sum < self.player_sum:
                return self._get_obs(), 1, True, False, self._get_info()
            else:
                return self._get_obs(), 0, True, False, self._get_info()
            
    def render(self, mode="ansi"):
        if mode == "ansi":
            return self._render_text()
        else:
            raise NotImplementedError()
    
    def _render_text(self):
        print(f"Player sum: {self.player_sum}, Dealer card: {self.dealer_card}, Dealer sum: {self.dealer_sum}")
    