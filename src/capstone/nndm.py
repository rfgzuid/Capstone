import torch
import torch.nn as nn

from .settings import Env


class NNDM(nn.Sequential):
    """
    Neural Network Dynamical Model
    - input: tensor of size (M, N_state + N_action): M samples of (s,a) pairs
    - output: model tries to predict the following state, output tensor (M, N_state)

    The forward pass is used to predict the change in state (delta_state), which is then added to the original state
    This was found to improve generalization to unseen states

    Noise is added to the output of the NNDM to simulate a stochastic environment
    """

    def __init__(self, env: Env) -> None:
        self.env = env.env

        self.action_size = 1 if env.is_discrete else self.env.action_space.shape[0]
        self.observation_size = self.env.observation_space.shape[0]

        hidden_layers = env.settings['NNDM_layers']
        node_counts = [self.observation_size + self.action_size, *hidden_layers, self.observation_size]

        activation = env.settings['NNDM_activation']
        layers = []

        for idx in range(len(node_counts) - 1):
            layers.append(nn.Linear(node_counts[idx], node_counts[idx+1]))

            # don't add activation after the last linear layer
            if idx != len(node_counts) - 2:
                layers.append(activation())

        super(NNDM, self).__init__(*layers)

        self.criterion = env.settings['NNDM_criterion']
        self.optimizer = env.settings['NNDM_optim'](self.parameters(), lr=env.settings['NNDM_lr'])

    def update(self, batch):
        if batch is None:
            return

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)

        x_train = torch.cat([state_batch, action_batch], dim=1)
        y_train = torch.cat(batch.next_state)

        self.optimizer.zero_grad()

        y_pred = self(x_train)

        loss = self.criterion()(y_pred, y_train)
        loss.backward()

        self.optimizer.step()

        return loss.item()
