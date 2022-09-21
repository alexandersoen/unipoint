import os
import json
import torch
import torch.nn as nn

from abc import ABC, abstractmethod

from torch.distributions.uniform import Uniform

from models.abstract_neural import AbstractNeural

class AbstractSum(AbstractNeural, ABC):

    def __init__(self, recurrent_output_size, recurrent_num_layers,
            recurrent_kwarg, num_basis, num_param, compensator_sims):

        super().__init__(
            recurrent_output_size, recurrent_num_layers, recurrent_kwarg)

        self.mc_approx = True

        # Generate the recurrent to parameter layers
        self.weight_layer = nn.Linear(
            recurrent_output_size, num_basis * num_param)

        self.num_basis = num_basis
        self.num_param = num_param
        self.recurrent_num_layers = recurrent_num_layers
        self.recurrent_output_size = recurrent_output_size

        self.compensator_sims = compensator_sims

        hidden_size = (recurrent_num_layers, recurrent_output_size)
        self.learnable_hidden = nn.Parameter(torch.randn(hidden_size) * 0.01)

    def save_parameters(self, file_dir):
        save_dict = {
            'num_basis': self.num_basis,
            'num_param': self.num_param,
            'recurrent_output_size': self.recurrent_output_size,
            'recurrent_num_layers': self.recurrent_num_layers,
            'compensator_sims': self.compensator_sims,
            }

        file = os.path.join(file_dir, 'model_params.json')
        with open(file, 'w') as f:
            json.dump(save_dict, f)

    def load_parameters(self, file_dir):
        file = os.path.join(file_dir, 'model_params.json')
        with open(file, 'r') as f:
            load_dict = json.load(f)

        self.num_basis = load_dict['num_basis']
        self.num_param = load_dict['num_param']
        self.recurrent_output_size = load_dict['recurrent_output_size']
        self.recurrent_num_layers = load_dict['recurrent_num_layers']
        self.compensator_sims = load_dict['compensator_sims']

    def get_params(self, h):
        # calculate parameters
        weights = self.weight_layer(h)

        return torch.split(weights, self.num_basis, dim=-1)

    def ll(self, batch):
        h = self.get_hidden_state(batch)

        ts = batch.int_events  # (*)
        params = self.get_params(h)  # (*, num_basis)

        basis_vals = self.kernel(ts.unsqueeze(-1), params) # (*, num_basis)
        int_vals = self.to_positive(basis_vals.sum(-1))

        int_sum = torch.zeros(len(batch))
        compensator = torch.zeros(len(batch))
        for idx in range(len(batch)):
            cur_m = batch.mask[idx]

            # Intensity sum calculation
            cur_int = int_vals[idx][cur_m]
            cur_int_sum = torch.sum(torch.log(cur_int + 1e-8))

            int_sum[idx] = cur_int_sum

            if self.mc_approx:
                # Compensator calculation
                cur_ts = ts[idx][cur_m]
                mc_ts = cur_ts.repeat_interleave(self.compensator_sims) + 1e-8

                sampler = Uniform(torch.zeros_like(mc_ts), mc_ts)
                mc_points = sampler.sample()

                mc_params = [p[idx][cur_m, :].repeat_interleave(self.compensator_sims, dim=0).unsqueeze(0) for p in params]  # (masked * x sims, num_basis)

                mc_basis = self.kernel(mc_points.unsqueeze(-1), mc_params) # (masked * x sims, num_basis)
                mc_int = self.to_positive(mc_basis.sum(-1))  # (masked * x sims)

                mc_approx = mc_int.view(-1, self.compensator_sims).sum(-1) * cur_ts / self.compensator_sims

                mc_integral = torch.sum(mc_approx)

                compensator[idx] = mc_integral
            else:
                # Trapezoidal rule
                cur_ts = ts[idx][cur_m]
                compensator[idx] = torch.sum((cur_int[:-1] + cur_int[1:]) * cur_ts[1:] / 2)

        return (int_sum - compensator) / batch.seq_lengths

    def forward(self, batch):
        h = self.get_hidden_state(batch)

        ts = batch.int_events  # (*)
        params = self.get_params(h)  # (*, num_basis)

        basis_vals = self.kernel(ts.unsqueeze(-1), params) # (*, num_basis)
        int_vals = self.to_positive(basis_vals.sum(-1))

        intensity = []
        for idx in range(len(batch)):
            m = batch.mask[idx]
            intensity.append(int_vals[idx][m])

        return intensity

    def intensity_func(self, tau, hist, params):
        return self.to_positive(torch.sum(self.kernel(tau.unsqueeze(-1), params), dim=-1))

    def init_hidden(self, batch_size):

        hidden = self.learnable_hidden.expand(
            self.recurrent_num_layers,
            batch_size,
            self.recurrent_output_size
        )

        return hidden

    @abstractmethod
    def recurrent_builder(
            self, input_size, output_size, num_layers, kwargs):
        pass

    @abstractmethod
    def kernel(self, tau, weight):
        pass

    @abstractmethod
    def to_positive(self, x):
        pass
