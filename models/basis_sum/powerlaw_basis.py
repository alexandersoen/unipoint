import torch
import torch.nn as nn
import torch.nn.functional as F

from models.basis_sum.abstract_sum import AbstractSum

class PowerlawSum(AbstractSum):

    def __init__(
            self, rnn_output_size, rnn_num_layer, num_exp,
            compensator_sims=200):

        super().__init__(
            recurrent_output_size=rnn_output_size,
            recurrent_num_layers=rnn_num_layer,
            recurrent_kwarg={},
            num_basis=num_exp,
            num_param=2,
            compensator_sims=compensator_sims)

    def recurrent_builder(
            self, input_size, output_size, num_layers, kwargs):

        return nn.RNN(
            input_size=input_size,
            hidden_size=output_size,
            num_layers=num_layers,
            **kwargs)

    def kernel(self, tau, weights):
        alphas, betas = weights

        return alphas * (1 + tau) ** (- betas ** 2)

    def to_positive(self, x):
        s = 1 / 1.5
        return F.softplus(x, beta=s)