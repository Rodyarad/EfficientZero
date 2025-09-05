import torch

from core.config import BaseConfig
from core.utils import WarpFrame, EpisodicLifeEnv, TimeLimit
from core.dataset import Transforms
from .env_wrapper import Shapes2dWrapper, SlotExtractor, SlotExtractorWrapper
from .model import ObjectZero
from .shapes2d import Shapes2d
from . import register
from omegaconf import OmegaConf
from collections import namedtuple
from .ocr.slate.slate import SLATE
import gym


class Shapes2dConfig(BaseConfig):
    def __init__(self):
        super(Shapes2dConfig, self).__init__(
            training_steps=100000,
            last_steps=20000,
            test_interval=5000,
            log_interval=1,
            vis_interval=100,
            test_episodes=30,
            checkpoint_interval=100,
            target_model_interval=200,
            save_ckpt_interval=5000,
            max_moves=100,
            test_max_moves=100,
            history_length=400,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=50,
            batch_size=256,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=False,
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=100000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=2,
            total_transitions=100 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=0,
            stacked_observations=1,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            rnn_hidden_size=64,
            rnn_horizon_len=5,
            debug = False,
        )

        self.start_transitions = self.start_transitions * 1000
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.name_project = "objectzero"
        self.run_id = ""
        self.resume_path = ""

        self.ocr_config_path = 'config/shapes2d/ocr/slate/config/navigation5x5.yaml'
        self.checkpoint_path = 'config/shapes2d/ocr/slate_weights/navigation5х5.pth'
        self.num_slots = 6
        self.slot_dim = 64
        self.latent_dim = 512
        self.slot_based = True



    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        obs_shape = (self.image_channel, 64, 64)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])

        game = self.new_game()
        self.action_space_size = game.action_space.n

    def get_uniform_network(self):
        return ObjectZero(
            slot_dim=self.slot_dim,
            laten_dim=self.latent_dim,
            n_slots=self.num_slots,
            action_space_size=self.action_space_size,
            update_bias=-1,
            fc_reward_layers=self.resnet_fc_reward_layers,
            fc_value_layers=self.resnet_fc_value_layers,
            fc_policy_layers=self.resnet_fc_policy_layers,
            reward_support_size=self.reward_support.size,
            value_support_size=self.value_support.size,
            inverse_value_transform=self.inverse_value_transform,
            inverse_reward_transform=self.inverse_reward_transform,
            rnn_hidden_size=self.rnn_hidden_size,
            bn_mt=self.bn_mt,
            init_zero=self.init_zero,
            state_norm=self.state_norm
        )

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        env = gym.make(self.env_name)

        env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)
        #env = TimeLimit(env, max_episode_steps=self.max_moves)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers.record_video import RecordVideo
            env = RecordVideo(
                env,
                video_folder=save_path,
                episode_trigger=video_callable,
                name_prefix=f"video-{uid}"
            )

        config_ocr = OmegaConf.load(self.ocr_config_path)
        config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(64, 3)
        slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
        state_dict = torch.load(self.checkpoint_path)["ocr_module_state_dict"]
        slate._module.load_state_dict(state_dict)
        slate.requires_grad_(False)
        slate.eval()

        slot_extractor = SlotExtractor(model=slate, device='cuda', name_model='SLATE')

        env = SlotExtractorWrapper(env, slot_extractor, self.num_slots, self.slot_dim)
        return env

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


    def transform(self, images):
        return self.transforms.transform(images)


class Shapes2dTestConfig(BaseConfig):
    def __init__(self):
        super(Shapes2dTestConfig, self).__init__(
            training_steps=10000,
            last_steps=2000,
            test_interval=20,
            log_interval=1,
            vis_interval=20,
            test_episodes=2,
            checkpoint_interval=10,
            target_model_interval=20,
            save_ckpt_interval=10,
            max_moves=10,
            test_max_moves=10,
            history_length=5,
            discount=0.997,
            dirichlet_alpha=0.3,
            value_delta_max=0.01,
            num_simulations=2,
            batch_size=2,
            td_steps=5,
            num_actors=1,
            # network initialization/ & normalization
            episode_life=True,
            init_zero=True,
            clip_reward=False,
            # storage efficient
            cvt_string=False,
            image_based=False,
            # lr scheduler
            lr_warm_up=0.01,
            lr_init=0.2,
            lr_decay_rate=0.1,
            lr_decay_steps=100000,
            auto_td_steps_ratio=0.3,
            # replay window
            start_transitions=2,
            total_transitions=10 * 1000,
            transition_num=1,
            # frame skip & stack observation
            frame_skip=0,
            stacked_observations=1,
            # coefficient
            reward_loss_coeff=1,
            value_loss_coeff=0.25,
            policy_loss_coeff=1,
            consistency_coeff=2,
            # reward sum
            rnn_hidden_size=64,
            rnn_horizon_len=5,
            debug = True,
        )

        self.start_transitions = self.start_transitions
        self.start_transitions = max(1, self.start_transitions)

        self.bn_mt = 0.1
        self.blocks = 1  # Number of blocks in the ResNet
        self.resnet_fc_reward_layers = [32]  # Define the hidden layers in the reward head of the dynamic network
        self.resnet_fc_value_layers = [32]  # Define the hidden layers in the value head of the prediction network
        self.resnet_fc_policy_layers = [32]  # Define the hidden layers in the policy head of the prediction network
        self.downsample = True  # Downsample observations before representation network (See paper appendix Network Architecture)
        self.name_project = "objectzero"
        self.run_id = ""
        self.resume_path = ""

        self.ocr_config_path = 'config/shapes2d/ocr/slate/config/navigation5x5.yaml'
        self.checkpoint_path = 'config/shapes2d/ocr/slate_weights/navigation5х5.pth'
        self.num_slots = 6
        self.slot_dim = 64
        self.latent_dim = 512
        self.slot_based = True



    def visit_softmax_temperature_fn(self, num_moves, trained_steps):
        if self.change_temperature:
            if trained_steps < 0.5 * (self.training_steps):
                return 1.0
            elif trained_steps < 0.75 * (self.training_steps):
                return 0.5
            else:
                return 0.25
        else:
            return 1.0

    def set_game(self, env_name, save_video=False, save_path=None, video_callable=None):
        self.env_name = env_name
        obs_shape = (self.image_channel, 64, 64)
        self.obs_shape = (obs_shape[0] * self.stacked_observations, obs_shape[1], obs_shape[2])

        game = self.new_game()
        self.action_space_size = game.action_space.n

    def get_uniform_network(self):
        return ObjectZero(
            slot_dim=self.slot_dim,
            laten_dim=self.latent_dim,
            n_slots=self.num_slots,
            action_space_size=self.action_space_size,
            update_bias=-1,
            fc_reward_layers=self.resnet_fc_reward_layers,
            fc_value_layers=self.resnet_fc_value_layers,
            fc_policy_layers=self.resnet_fc_policy_layers,
            reward_support_size=self.reward_support.size,
            value_support_size=self.value_support.size,
            inverse_value_transform=self.inverse_value_transform,
            inverse_reward_transform=self.inverse_reward_transform,
            rnn_hidden_size=self.rnn_hidden_size,
            bn_mt=self.bn_mt,
            init_zero=self.init_zero,
            state_norm=self.state_norm
        )

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        env = gym.make(self.env_name)

        env = WarpFrame(env, width=self.obs_shape[1], height=self.obs_shape[2], grayscale=self.gray_scale)
        env = TimeLimit(env, max_episode_steps=self.max_moves)

        if seed is not None:
            env.seed(seed)

        if save_video:
            from gym.wrappers.record_video import RecordVideo
            env = RecordVideo(
                env,
                video_folder=save_path,
                episode_trigger=video_callable,
                name_prefix=f"video-{uid}"
            )

        config_ocr = OmegaConf.load(self.ocr_config_path)
        config_env = namedtuple('EnvConfig', ['obs_size', 'obs_channels'])(64, 3)
        slate = SLATE(config_ocr, config_env, observation_space=None, preserve_slot_order=True)
        state_dict = torch.load(self.checkpoint_path)["ocr_module_state_dict"]
        slate._module.load_state_dict(state_dict)
        slate.requires_grad_(False)
        slate.eval()

        slot_extractor = SlotExtractor(model=slate, device='cuda', name_model='SLATE')

        env = SlotExtractorWrapper(env, slot_extractor, self.num_slots, self.slot_dim)
        return env

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def transform(self, images):
        return self.transforms.transform(images)


game_config = Shapes2dConfig()
game_test_config = Shapes2dTestConfig()