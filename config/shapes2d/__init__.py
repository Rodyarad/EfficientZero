import torch

from core.config import BaseConfig
from core.utils import WarpFrame, EpisodicLifeEnv, TimeLimit
from core.dataset import Transforms
from .env_wrapper import Shapes2dWrapper
from .model import EfficientZeroNet
from .shapes2d import Shapes2d
from . import register
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
            lstm_hidden_size=512,
            lstm_horizon_len=5,
            # siamese
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            debug=False
        )

        self.start_transitions = self.start_transitions * 1000
        self.start_transitions = max(1, self.start_transitions)
        self.blocks = 2
        self.hidden_shape = 128
        self.rep_net_shape = 256
        self.dyn_shape = 256
        self.rew_net_shape = [256, 256]
        self.val_net_shape = [256, 256]
        self.pi_net_shape = [256, 256]
        self.use_bn = True
        self.use_p_norm = False
        self.noisy_net = False
        self.name_project = "objectzero"
        self.run_id = ""
        self.resume_path = ""

        self.coord_based = True


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
        self.obs_shape = 10 * self.stacked_observations

        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            self.hidden_shape,
            self.rep_net_shape,
            self.dyn_shape,
            self.rew_net_shape,
            self.val_net_shape,
            self.pi_net_shape,
            self.use_p_norm,
            self.use_bn,
            self.noisy_net,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        env = gym.make(self.env_name)
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
        return Shapes2dWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)



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
            lstm_hidden_size=512,
            lstm_horizon_len=5,
            # siamese
            proj_hid=1024,
            proj_out=1024,
            pred_hid=512,
            pred_out=1024,
            debug=True
        )

        self.start_transitions = self.start_transitions * 1000
        self.start_transitions = max(1, self.start_transitions)
        self.blocks = 2
        self.hidden_shape = 128
        self.rep_net_shape = 256
        self.dyn_shape = 256
        self.rew_net_shape = [256, 256]
        self.val_net_shape = [256, 256]
        self.pi_net_shape = [256, 256]
        self.use_bn = True
        self.use_p_norm = False
        self.noisy_net = False
        self.name_project = "objectzero"
        self.run_id = ""
        self.resume_path = ""

        self.coord_based = True


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
        self.obs_shape = 10 * self.stacked_observations

        game = self.new_game()
        self.action_space_size = game.action_space_size

    def get_uniform_network(self):
        return EfficientZeroNet(
            self.obs_shape,
            self.action_space_size,
            self.blocks,
            self.reward_support.size,
            self.value_support.size,
            self.inverse_value_transform,
            self.inverse_reward_transform,
            self.lstm_hidden_size,
            self.hidden_shape,
            self.rep_net_shape,
            self.dyn_shape,
            self.rew_net_shape,
            self.val_net_shape,
            self.pi_net_shape,
            self.use_p_norm,
            self.use_bn,
            self.noisy_net,
            proj_hid=self.proj_hid,
            proj_out=self.proj_out,
            pred_hid=self.pred_hid,
            pred_out=self.pred_out,
            init_zero=self.init_zero,
            state_norm=self.state_norm)

    def new_game(self, seed=None, save_video=False, save_path=None, video_callable=None, uid=None, test=False, final_test=False):
        env = gym.make(self.env_name)
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
        return Shapes2dWrapper(env, discount=self.discount, cvt_string=self.cvt_string)

    def scalar_reward_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)

    def scalar_value_loss(self, prediction, target):
        return -(torch.log_softmax(prediction, dim=1) * target).sum(1)


game_config = Shapes2dConfig()
game_test_config = Shapes2dTestConfig()