import math
import torch

import numpy as np
import torch.nn as nn
import torchrl

from core.model import BaseNet, renormalize


class PNorm(nn.Module):
    def __init__(self, eps=1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        assert len(x.shape) == 2
        return nn.functional.normalize(x, dim=1, eps=self.eps)

class RunningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-5, momentum=0.1):
        super(RunningMeanStd, self).__init__()
        self.epsilon = epsilon
        self.momentum = momentum
        self.count = 1e3
        self.register_buffer('running_mean', torch.zeros(shape))
        self.register_buffer('running_var', torch.ones(shape))

    def forward(self, x):
        if self.training:
            mean = x.mean(dim=0)
            var = x.var(dim=0, unbiased=False)
            batch_count = x.shape[0]
            self.running_mean, self.running_var, self.count = self.update_mean_var_count_from_moments(self.running_mean, self.running_var, self.count, mean, var, batch_count)
            global_mean = self.running_mean
            global_var = self.running_var
        else:
            global_mean = self.running_mean
            global_var = self.running_var
        x = (x - global_mean) / torch.sqrt(global_var + self.epsilon)
        return x

    def update_mean_var_count_from_moments(self, mean, var, count, batch_mean, batch_var, batch_count):
        """Updates the mean, var and count using the previous mean, var, count and batch values."""
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count

def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    init_zero=False,
    use_bn=True,
    p_norm=False,
    noisy=False
):
    """MLP layers
    Parameters
    ----------
    input_size: int
        dim of inputs
    layer_sizes: list
        dim of hidden layers
    output_size: int
        dim of outputs
    init_zero: bool
        zero initialization for the last layer (including w and b).
        This can provide stable zero outputs in the beginning.
    """
    sizes = [input_size] + layer_sizes + [output_size]
    layers = []
    for i in range(len(sizes) - 1):
        if i < len(sizes) - 2:
            act = activation
            if use_bn:
                layers += [
                    torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                    nn.BatchNorm1d(sizes[i + 1]),
                    act()
                ]
            else:
                layers += [torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                           act()]
        else:
            if p_norm == True:
                layers += [PNorm()]
            act = output_activation
            layers += [torchrl.modules.NoisyLinear(sizes[i], sizes[i + 1], std_init=0.5) if noisy else nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        if noisy:
            layers[-2].reset_parameters()
        else:
            layers[-2].weight.data.fill_(0)
            layers[-2].bias.data.fill_(0)


    return nn.Sequential(*layers)


# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, input_shape, hidden_shape):
        super(ResidualBlock, self).__init__()
        self.ln1 = nn.LayerNorm(input_shape)
        self.linear1 = nn.Linear(input_shape, hidden_shape)
        self.linear2 = nn.Linear(hidden_shape, input_shape)

    def forward(self, x):
        identity = x
        out = self.ln1(x)
        out = self.linear1(out)
        out = nn.functional.relu(out)
        out = self.linear2(out)

        out += identity
        return out


# Encode the observations into hidden states
class RepresentationNetwork(nn.Module):
    def __init__(
            self,
            observation_shape,
            num_blocks,
            rep_net_shape,
            hidden_shape,
    ):
        """Representation network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        num_blocks: int
            number of res blocks
        rep_net_shape: int
            shape of hidden layers
        hidden_shape:
            dim of output hidden state
        use_bn: bool
            True -> Batch normalization
        """
        super().__init__()

        self.running_mean_std = RunningMeanStd(observation_shape)
        self.mlp = nn.Linear(observation_shape, hidden_shape)
        self.ln = nn.LayerNorm(hidden_shape)
        self.Rep_resblocks = nn.ModuleList(
            [ResidualBlock(hidden_shape, rep_net_shape) for _ in range(num_blocks)]
        )

    def forward(self, x):

        x = self.running_mean_std(x)
        x = self.mlp(x)
        x = self.ln(x)
        x = torch.tanh(x)
        # res block
        for block in self.Rep_resblocks:
            x = block(x)

        return x

    def get_param_mean(self):
        mean = []
        for name, param in self.named_parameters():
            mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        mean = sum(mean) / len(mean)
        return mean


# Predict next hidden states given current states and actions
class DynamicsNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        action_shape,
        num_blocks,
        dyn_shape,
        rew_net_shape,
        reward_support_size,
        rnn_hidden_size,
        init_zero=False,
        use_bn=True,
    ):
        """Dynamics network
        Parameters
        ----------
        hidden_shape: int
            dim of input hidden state
        action_shape: int
            dim of action
        num_blocks: int
            number of res blocks
        dyn_shape: int
            number of nodes of hidden layer
        act_embed_shape: int
            dim of action embedding
        rew_net_shape: list
            hidden layers of the reward prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        init_zero: bool
            True -> zero initialization for the last layer of reward mlp
        use_bn: bool
            True -> Batch normalization
        """
        super().__init__()
        self.hidden_shape = hidden_shape

        self.dyn_ln_1 = nn.LayerNorm(hidden_shape + 1)
        self.dyn_net_1 = nn.Linear(hidden_shape + 1, dyn_shape)

        self.dyn_ln_2 = nn.LayerNorm(dyn_shape)
        self.dyn_net_2 = nn.Linear(dyn_shape, hidden_shape)

        if num_blocks > 0:
            self.dyn_resblocks = nn.ModuleList(
                [ResidualBlock(hidden_shape, dyn_shape) for _ in range(num_blocks)]
            )
        else:
            self.dyn_resblocks = nn.ModuleList([])

        self.rew_net_shape = rew_net_shape
        self.reward_support_size = reward_support_size
        self.rew_resblock = ResidualBlock(self.hidden_shape, self.hidden_shape)
        self.ln = nn.LayerNorm(self.hidden_shape)
        self.lstm = nn.LSTM(input_size=self.hidden_shape, hidden_size=rnn_hidden_size)
        self.rew_net = mlp(rnn_hidden_size, self.rew_net_shape, self.reward_support_size,
                           init_zero=init_zero,
                           use_bn=use_bn)


    def forward(self, hidden, reward_hidden=None):
        # imporved res block 1st
        state_no_act = hidden[:, :-1]
        x = self.dyn_ln_1(hidden)
        x = self.dyn_net_1(x)
        x = nn.functional.relu(x)
        x = self.dyn_net_2(x)

        state = state_no_act + x

        # residual tower for dynamic model (2nd -> num blocks)
        for block in self.dyn_resblocks:
            state = block(state)

        next_state = self.rew_resblock(state)
        next_state = self.ln(next_state)
        next_state = next_state.unsqueeze(0)
        reward, reward_hidden = self.lstm(next_state, reward_hidden)
        reward = reward.squeeze(0)
        reward = self.rew_net(reward)


        return state, reward_hidden, reward

    def get_dynamic_mean(self):
        dynamic_mean = np.abs(self.conv.weight.detach().cpu().numpy().reshape(-1)).tolist()

        for block in self.resblocks:
            for name, param in block.named_parameters():
                dynamic_mean += np.abs(param.detach().cpu().numpy().reshape(-1)).tolist()
        dynamic_mean = sum(dynamic_mean) / len(dynamic_mean)
        return dynamic_mean

    def get_reward_mean(self):
        reward_w_dist = self.conv1x1_reward.weight.detach().cpu().numpy().reshape(-1)

        for name, param in self.fc.named_parameters():
            temp_weights = param.detach().cpu().numpy().reshape(-1)
            reward_w_dist = np.concatenate((reward_w_dist, temp_weights))
        reward_mean = np.abs(reward_w_dist).mean()
        return reward_w_dist, reward_mean


# predict the value and policy given hidden states
class PredictionNetwork(nn.Module):
    def __init__(
        self,
        hidden_shape,
        val_net_shape,
        pi_net_shape,
        action_shape,
        full_support_size,
        init_zero=False,
        use_bn=True,
        p_norm=False,
        policy_distr='squashed_gaussian',
        noisy=False,
        value_support=None,
        **kwargs
    ):
        super().__init__()
        self.hidden_shape = hidden_shape
        self.val_net_shape = val_net_shape
        self.action_shape = action_shape
        self.pi_net_shape = pi_net_shape

        self.action_space_size = action_shape
        self.init_std = 1.0
        self.min_std = 0.1
        self.policy_distr = policy_distr

        self.val_resblock = ResidualBlock(hidden_shape, hidden_shape)
        self.pi_resblock = ResidualBlock(hidden_shape, hidden_shape)

        self.val_ln = nn.LayerNorm(hidden_shape)
        self.pi_ln = nn.LayerNorm(hidden_shape)


        self.val_net = mlp(self.hidden_shape, self.val_net_shape, full_support_size, use_bn=use_bn)
        self.pi_net = mlp(self.hidden_shape, self.pi_net_shape, self.action_shape * 2,
                          init_zero=init_zero,
                          use_bn=use_bn,
                          p_norm=p_norm,
                          noisy=noisy)

        self.noisy = noisy
        self.value_support = value_support

    def reset_noise(self):
        if self.noisy:
            for layer in self.pi_net:
                try:
                    layer.reset_noise()
                except:
                    pass

    def forward(self, x):
        value = self.val_resblock(x)
        value = self.val_ln(value)
        values = []
        values.append(self.val_net(value))
        values = torch.stack(values)

        policy = self.pi_resblock(x)
        policy = self.pi_ln(policy)
        policy = self.pi_net(policy)

        action_space_size = policy.shape[-1] // 2
        if self.policy_distr == 'squashed_gaussian':
            policy[:, :action_space_size] = 5 * torch.tanh(policy[:, :action_space_size] / 5)  # soft clamp mu
            policy[:, action_space_size:] = torch.nn.functional.softplus(
                policy[:, action_space_size:] + self.init_std) + self.min_std  # same as Dreamer-v3

        return policy, values

    def log_std(self, x, low, dif):
        return low + 0.5 * dif * (torch.tanh(x) + 1)


class EfficientZeroNet(BaseNet):
    def __init__(
        self,
        observation_shape,
        action_space_size,
        num_blocks,
        reward_support_size,
        value_support_size,
        inverse_value_transform,
        inverse_reward_transform,
        lstm_hidden_size,
        hidden_shape,
        rep_net_shape,
        dyn_shape,
        rew_net_shape,
        val_net_shape,
        pi_net_shape,
        use_p_norm,
        use_bn,
        noisy_net,
        proj_hid=256,
        proj_out=256,
        pred_hid=64,
        pred_out=256,
        init_zero=False,
        state_norm=False,
        policy_distribution = "squashed_gaussian"
    ):
        """EfficientZero network
        Parameters
        ----------
        observation_shape: tuple or list
            shape of observations: [C, W, H]
        action_space_size: int
            action space
        num_blocks: int
            number of res blocks
        num_channels: int
            channels of hidden states
        reduced_channels_reward: int
            channels of reward head
        reduced_channels_value: int
            channels of value head
        reduced_channels_policy: int
            channels of policy head
        fc_reward_layers: list
            hidden layers of the reward prediction head (MLP head)
        fc_value_layers: list
            hidden layers of the value prediction head (MLP head)
        fc_policy_layers: list
            hidden layers of the policy prediction head (MLP head)
        reward_support_size: int
            dim of reward output
        value_support_size: int
            dim of value output
        downsample: bool
            True -> do downsampling for observations. (For board games, do not need)
        inverse_value_transform: Any
            A function that maps value supports into value scalars
        inverse_reward_transform: Any
            A function that maps reward supports into value scalars
        lstm_hidden_size: int
            dim of lstm hidden
        bn_mt: float
            Momentum of BN
        proj_hid: int
            dim of projection hidden layer
        proj_out: int
            dim of projection output layer
        pred_hid: int
            dim of projection head (prediction) hidden layer
        pred_out: int
            dim of projection head (prediction) output layer
        init_zero: bool
            True -> zero initialization for the last layer of value/policy mlp
        state_norm: bool
            True -> normalization for hidden states
        """
        super(EfficientZeroNet, self).__init__(inverse_value_transform, inverse_reward_transform, lstm_hidden_size)
        self.hidden_shape = hidden_shape
        self.proj_hid = proj_hid
        self.proj_out = proj_out
        self.pred_hid = pred_hid
        self.pred_out = pred_out
        self.init_zero = init_zero
        self.state_norm = state_norm
        self.action_space_size = action_space_size

        self.representation_network = RepresentationNetwork(
            observation_shape,
            num_blocks,
            rep_net_shape,
            self.hidden_shape,
        )

        self.dynamics_network = DynamicsNetwork(
            self.hidden_shape,
            self.action_space_size,
            num_blocks,
            dyn_shape,
            rew_net_shape,
            reward_support_size,
            rnn_hidden_size=lstm_hidden_size,
            init_zero=init_zero,
            use_bn=use_bn,
        )

        self.prediction_network = PredictionNetwork(
            self.hidden_shape,
            val_net_shape,
            pi_net_shape,
            self.action_space_size,
            value_support_size,
            init_zero=self.init_zero,
            use_bn=use_bn,
            p_norm=use_p_norm,
            policy_distr=policy_distribution,
            noisy=noisy_net
        )

        # projection
        self.projection = nn.Sequential(
            nn.Linear(self.hidden_shape, self.proj_hid),
            nn.LayerNorm(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_hid),
            nn.LayerNorm(self.proj_hid),
            nn.ReLU(),
            nn.Linear(self.proj_hid, self.proj_out),
            nn.LayerNorm(self.proj_out)
        )

        self.projection_head = nn.Sequential(
            nn.Linear(self.proj_out, self.pred_hid),
            nn.LayerNorm(self.pred_hid),
            nn.ReLU(),
            nn.Linear(self.pred_hid, self.pred_out),
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def representation(self, observation):
        encoded_state = self.representation_network(observation)
        if not self.state_norm:
            return encoded_state
        else:
            encoded_state_normalized = renormalize(encoded_state)
            return encoded_state_normalized

    def dynamics(self, encoded_state, reward_hidden, action):
        # Stack encoded_state with a game specific one hot encoded action
        action_one_hot = (
            torch.ones(
                (
                    encoded_state.shape[0],
                    1,
                )
            )
            .to(action.device)
            .float()
        )
        action_one_hot = action * action_one_hot / self.action_space_size
        x = torch.cat((encoded_state, action_one_hot), dim=1)
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(x, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

    def get_params_mean(self):
        representation_mean = self.representation_network.get_param_mean()
        dynamic_mean = self.dynamics_network.get_dynamic_mean()
        reward_w_dist, reward_mean = self.dynamics_network.get_reward_mean()

        return reward_w_dist, representation_mean, dynamic_mean, reward_mean

    def project(self, hidden_state, with_grad=True):
        # only the branch of proj + pred can share the gradients
        hidden_state = hidden_state.view(-1, self.hidden_shape)
        proj = self.projection(hidden_state)

        # with grad, use proj_head
        if with_grad:
            proj = self.projection_head(proj)
            return proj
        else:
            return proj.detach()

