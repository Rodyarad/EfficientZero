import math
import torch

import numpy as np
import torch.nn as nn

from core.model import BaseNet, renormalize
from core.utils import to_one_hot, make_node_mlp_layers, unsorted_segment_sum


def mlp(
    input_size,
    layer_sizes,
    output_size,
    output_activation=nn.Identity,
    activation=nn.ReLU,
    momentum=0.1,
    init_zero=False,
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
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       nn.BatchNorm1d(sizes[i + 1], momentum=momentum),
                       act()]
        else:
            act = output_activation
            layers += [nn.Linear(sizes[i], sizes[i + 1]),
                       act()]

    if init_zero:
        layers[-2].weight.data.fill_(0)
        layers[-2].bias.data.fill_(0)

    return nn.Sequential(*layers)

class GNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, action_dim, num_objects, ignore_action=False, copy_action=False,
                 act_fn='relu', layer_norm=True, num_layers=3, use_interactions=True, edge_actions=False,
                 output_dim=None):
        super(GNN, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        if self.output_dim is None:
            self.output_dim = self.input_dim

        self.num_objects = num_objects
        self.ignore_action = ignore_action
        self.copy_action = copy_action
        self.use_interactions = use_interactions
        self.edge_actions = edge_actions
        self.num_layers = num_layers

        if self.ignore_action:
            self.action_dim = 0
        else:
            self.action_dim = action_dim

        tmp_action_dim = self.action_dim
        edge_mlp_input_size = self.input_dim * 2 + int(self.edge_actions) * tmp_action_dim

        self.edge_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            edge_mlp_input_size, self.hidden_dim, act_fn, layer_norm
        ))

        if self.num_objects == 1 or not self.use_interactions:
            node_input_dim = self.input_dim + tmp_action_dim
        else:
            node_input_dim = hidden_dim + self.input_dim + tmp_action_dim

        self.node_mlp = nn.Sequential(*self.make_node_mlp_layers_(
            node_input_dim, self.output_dim, act_fn, layer_norm
        ))

        self.edge_list = None
        self.batch_size = 0

    def _edge_model(self, source, target, action=None):
        if action is None:
            x = [source, target]
        else:
            x = [source, target, action]

        out = torch.cat(x, dim=1)
        return self.edge_mlp(out)

    def _node_model(self, node_attr, edge_index, edge_attr):
        if edge_attr is not None:
            row, col = edge_index
            agg = unsorted_segment_sum(
                edge_attr, row, num_segments=node_attr.size(0))
            out = torch.cat([node_attr, agg], dim=1)
        else:
            out = node_attr
        return self.node_mlp(out)

    def _get_edge_list_fully_connected(self, batch_size, num_objects, device):
        # Only re-evaluate if necessary (e.g. if batch size changed).
        if self.edge_list is None or self.batch_size != batch_size:
            self.batch_size = batch_size

            # Create fully-connected adjacency matrix for single sample.
            adj_full = torch.ones(num_objects, num_objects)

            # Remove diagonal.
            adj_full -= torch.eye(num_objects)
            self.edge_list = adj_full.nonzero()

            # Copy `batch_size` times and add offset.
            self.edge_list = self.edge_list.repeat(batch_size, 1)
            offset = torch.arange(
                0, batch_size * num_objects, num_objects).unsqueeze(-1)
            offset = offset.expand(batch_size, num_objects * (num_objects - 1))
            offset = offset.contiguous().view(-1)
            self.edge_list += offset.unsqueeze(-1)

            # Transpose to COO format -> Shape: [2, num_edges].
            self.edge_list = self.edge_list.transpose(0, 1)
            self.edge_list = self.edge_list.to(device)

        return self.edge_list

    def process_action_(self, action):
        if self.copy_action:
            if action.shape[1] == 1 and (action.dtype in (torch.int32, torch.int64)):
                # action is an integer
                action = action.squeeze(1)
                action_vec = to_one_hot(action, self.action_dim).repeat(1, self.num_objects)
            else:
                # action is a vector
                action_vec = action.repeat(1, self.num_objects)
            # mix node and batch dimension
            action_vec = action_vec.reshape(-1, self.action_dim).float()
        else:
            # we have a separate action for each node
            if action.shape[1] == 1 and (action.dtype in (torch.int32, torch.int64)):
                # index for both object and action
                action = action.squeeze(1)
                action_vec = to_one_hot(action, self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)
            else:
                action_vec = action.reshape(action.size(0), self.action_dim * self.num_objects)
                action_vec = action_vec.reshape(-1, self.action_dim)

        return action_vec

    def forward(self, states, action):

        device = states.device
        batch_size = states.size(0)
        num_nodes = states.size(1)

        # states: [batch_size (B), num_objects, embedding_dim]
        # node_attr: Flatten states tensor to [B * num_objects, embedding_dim]
        node_attr = states.reshape(-1, self.input_dim)

        action_vec = None
        if not self.ignore_action:
            action_vec = self.process_action_(action)

        edge_attr = None
        edge_index = None

        if num_nodes > 1 and self.use_interactions:
            # edge_index: [B * (num_objects*[num_objects-1]), 2] edge list
            edge_index = self._get_edge_list_fully_connected(
                batch_size, num_nodes, device)

            row, col = edge_index
            edge_attr = self._edge_model(node_attr[row], node_attr[col], action_vec[row] if self.edge_actions else None)

        if not self.ignore_action:
            # Attach action to each state
            node_attr = torch.cat([node_attr, action_vec], dim=-1)

        node_attr = self._node_model(
            node_attr, edge_index, edge_attr)

        # [batch_size, num_nodes, hidden_dim]
        node_attr = node_attr.view(batch_size, num_nodes, -1)

        return node_attr

    def make_node_mlp_layers_(self, input_dim, output_dim, act_fn, layer_norm):
        return make_node_mlp_layers(self.num_layers, input_dim, self.hidden_dim, output_dim, act_fn, layer_norm)


class OCDynamicsNetwork(nn.Module):
    def __init__(
        self,
        slot_dim,
        latent_dim,
        action_space_size,
        n_slots,
        update_bias,
        fc_reward_layers,
        full_support_size,
        rnn_hidden_size=64,
        momentum=0.1,
        init_zero=False,
    ):
        super().__init__()

        self.slot_dim = slot_dim
        self.latent_dim = latent_dim
        self.action_space_size = action_space_size
        self.n_slots = n_slots
        self.gnn_dyn = GNN(input_dim=self.slot_dim, hidden_dim=self.latent_dim,
                              action_dim=self.action_space_size, num_objects=self.n_slots, ignore_action=False,
                              copy_action=True, edge_actions=True)

        self.rnn_hidden_size = rnn_hidden_size
        self.act = nn.Tanh()
        self.update_bias = update_bias
        self.gnn_prefix = GNN(self.slot_dim + self.rnn_hidden_size, hidden_dim=self.latent_dim, action_dim=0, num_objects=self.n_slots,
                        ignore_action=True, copy_action=False, edge_actions=False, output_dim=3 * self.rnn_hidden_size)
        self.fc = mlp(rnn_hidden_size, fc_reward_layers, full_support_size, init_zero=init_zero, momentum=momentum)

    def forward(self, x, action, reward_hidden):
        state = self.gnn_dyn(x, action)

        hidden = reward_hidden.squeeze(0)
        full_state = torch.cat([x, hidden], dim=-1)
        parts = self.gnn_prefix(full_state, None)[0]
        reset, cand, update = torch.split(parts, [self.rnn_hidden_size] * 3, dim=-1)
        reset = torch.sigmoid(reset)
        cand = self.act(reset * cand)
        update = torch.sigmoid(update + self.update_bias)
        hidden = (update * cand + (1 - update) * x)
        value_prefix = self.fc(hidden.sum(dim=1))

        return state, reward_hidden.unsqueeze(0), value_prefix

class OCPredictionNetwork(nn.Module):
    def __init__(
        self,
        slot_dim,
        latent_dim,
        n_slots,
        action_space_size,
        fc_value_layers,
        fc_policy_layers,
        full_support_size,
        momentum=0.1,
        init_zero=False,
    ):
        super().__init__()
        self.slot_dim = slot_dim
        self.latent_dim = latent_dim
        self.n_slots = n_slots
        print()
        self.gnn_policy = GNN(input_dim=self.slot_dim, hidden_dim=self.latent_dim, action_dim=0,
                              num_objects=self.n_slots, ignore_action=True, copy_action=False, edge_actions=False)
        self.gnn_values = GNN(input_dim=self.slot_dim, hidden_dim=self.latent_dim, action_dim=0,
                               num_objects=self.n_slots, ignore_action=True, copy_action=False, edge_actions=False)
        self.fc_values = mlp(self.slot_dim, fc_value_layers, full_support_size,init_zero= init_zero, momentum = momentum)
        self.fc_policy = mlp(self.slot_dim, fc_policy_layers, action_space_size, init_zero=init_zero, momentum = momentum)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        policy = self.gnn_policy(x, action=None)
        policy = self.act(policy)
        policy = self.fc_policy(policy.sum(dim=1))

        value = self.gnn_values(x, action=None)
        value = self.act(value)
        value = self.fc_values(value.sum(dim=1))
        return policy, value

class ObjectZero(BaseNet):
    def __init__(
        self,
        slot_dim,
        laten_dim,
        n_slots,
        action_space_size,
        update_bias,
        fc_reward_layers,
        fc_value_layers,
        fc_policy_layers,
        reward_support_size,
        value_support_size,
        inverse_value_transform,
        inverse_reward_transform,
        rnn_hidden_size,
        bn_mt=0.1,
        init_zero=False,
        state_norm=False
    ):
        super(ObjectZero, self).__init__(inverse_value_transform, inverse_reward_transform, rnn_hidden_size)
        self.init_zero = init_zero
        self.state_norm = state_norm
        self.action_space_size = action_space_size
        self.n_slots = n_slots

        self.dynamics_network = OCDynamicsNetwork(
            slot_dim,
            laten_dim,
            action_space_size,
            n_slots,
            update_bias,
            fc_reward_layers,
            reward_support_size,
            rnn_hidden_size=rnn_hidden_size,
            momentum=bn_mt,
            init_zero=init_zero,
        )

        self.prediction_network = OCPredictionNetwork(
            slot_dim,
            laten_dim,
            n_slots,
            action_space_size,
            fc_value_layers,
            fc_policy_layers,
            value_support_size,
            momentum=bn_mt,
            init_zero=init_zero,
        )

    def prediction(self, encoded_state):
        policy, value = self.prediction_network(encoded_state)
        return policy, value

    def dynamics(self, encoded_state, reward_hidden, action):
        next_encoded_state, reward_hidden, value_prefix = self.dynamics_network(encoded_state, action, reward_hidden)

        if not self.state_norm:
            return next_encoded_state, reward_hidden, value_prefix
        else:
            next_encoded_state_normalized = renormalize(next_encoded_state)
            return next_encoded_state_normalized, reward_hidden, value_prefix

