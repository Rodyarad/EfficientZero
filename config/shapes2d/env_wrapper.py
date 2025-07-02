import numpy as np
from core.game import Game
from core.utils import arr_to_str, obs_to_tensor
import gym


class Shapes2dWrapper(Game):
    def __init__(self, env, discount: float, cvt_string=True):
        """Atari Wrapper
        Parameters
        ----------
        env: Any
            another env wrapper
        discount: float
            discount of env
        cvt_string: bool
            True -> convert the observation into string in the replay buffer
        """
        super().__init__(env, env.action_space.n, discount)
        self.cvt_string = cvt_string

    def legal_actions(self):
        return [_ for _ in range(self.env.action_space.n)]

    def get_max_episode_steps(self):
        return self.env.get_max_episode_steps()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation, reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        observation = observation.astype(np.uint8)

        if self.cvt_string:
            observation = arr_to_str(observation)

        return observation

    def close(self):
        self.env.close()


class SlotExtractor:
    def __init__(self, model, device, name_model):
        self._model = model
        self._device = device
        self._model.to(device)
        self.name_model = name_model

    def __call__(self, images, prev_slots, to_numpy=True):
        if len(images.shape) == 3:
            batch_images = images[np.newaxis, ...]
        else:
            batch_images = images

        if prev_slots is not None and len(prev_slots.shape) == 2:
            batch_prev_slots = prev_slots[np.newaxis, ...]
        else:
            batch_prev_slots = prev_slots

        batch_images = obs_to_tensor(batch_images, self._device)
        if batch_prev_slots is not None:
            batch_prev_slots = obs_to_tensor(batch_prev_slots, self._device)

        if self.name_model == 'SLATE':
            slots = self._model._module._get_slots(batch_images, prev_slots=batch_prev_slots).detach()
        else:
            slots = self._model(batch_images, prev_slots=batch_prev_slots).detach()

        if len(images.shape) == 3:
            slots = slots[0]

        if to_numpy:
            slots = slots.cpu().numpy()

        return slots

class SlotExtractorWrapper(gym.Wrapper):
    """
    Wrapper uses SlotExtractor in order to extract slots from the input image.
    """

    def __init__(self, env, slot_extractor, num_slots, slot_dim):
        super().__init__(env)

        self.slot_extractor = slot_extractor
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(num_slots, slot_dim), dtype=np.float32
        )
        self.prev_slots = None

    def _get_slots(self, frame, prev_slots=None):
        if prev_slots is None:
            prev_slots = self.slot_extractor(frame, prev_slots=None)

        return self.slot_extractor(frame, prev_slots=prev_slots)

    def reset(self):
        frame = self.env.reset()
        self.prev_slots = self._get_slots(frame, prev_slots=None)
        return self.prev_slots.copy()

    def step(self, action):
        frame, reward, done, info = self.env.step(action)
        self.prev_slots = self._get_slots(frame, prev_slots=self.prev_slots)
        return self.prev_slots.copy(), reward, done, info