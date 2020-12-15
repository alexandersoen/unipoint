#%%
import os
import json
import errno
import torch
import numpy as np

from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

from models.traditional.sequence_generators import generate_hawkes
from models.traditional.abstract_traditional import AbstractTraditional
from models.data import GeneralDataset, collate_no_marks

def loglikelihood_seq(data):
    seq, params = data
    mu, alpha, beta, delta = params

    max_t = seq[-1]
    seq = seq[:-1]

    cur_ll = 0
    for i in range(len(seq)):
        t = seq[i]
        history = seq[:i]

        kernel_vals = ((delta + t - h) ** (-(1 + beta)) for h in history)
        intensity = mu + alpha * sum(kernel_vals)

        cur_ll += np.log(intensity + 1e-9)

    cur_ll -= mu * max_t
    cur_ll += alpha / beta * sum((delta + max_t - t) ** (-beta) for t in seq)
    cur_ll -= alpha / beta * delta ** (-beta) * len(seq)

    return cur_ll

def jacobian_seq(data):
    seq, params = data
    mu, alpha, beta, delta = params

    max_t = seq[-1]
    seq = seq[:-1]

    cur_mu_deriv = 0
    cur_alpha_deriv = 0
    cur_beta_deriv = 0

    for i in range(len(seq)):
        t = seq[i]
        history = seq[:i]

        kernel_vals = ((delta + t - h) ** (-(1 + beta)) for h in history)
        intensity = mu + alpha * sum(kernel_vals)

        cur_mu_deriv += 1 / (intensity + 1e-9)

        cur_alpha_deriv += sum((delta + t - h) ** (-(1 + beta))  / intensity for h in history)
        cur_alpha_deriv += (delta + max_t - t) ** (-beta) / beta

        cur_beta_deriv -= alpha * sum((delta + t - h) ** (-1 - beta) * np.log(delta + t - h) / intensity for h in history)
        cur_beta_deriv -= alpha * (delta + max_t - t) ** (-beta) * (beta * np.log(delta + max_t - t) + 1) / (beta ** 2)

    cur_mu_deriv -= max_t

    cur_alpha_deriv -= len(seq) / beta * delta ** (-beta)

    cur_beta_deriv += len(seq) * alpha * delta ** (-beta) * (beta * np.log(delta) + 1) / (beta ** 2)

    cur_deriv = np.array([cur_mu_deriv, cur_alpha_deriv, cur_beta_deriv])

    return cur_deriv

class PowerlawHawkes(AbstractTraditional):

    def __init__(self, baseline, multiplier, exponent, cutoff):
        self.baseline = baseline
        self.multiplier = multiplier
        self.exponent = exponent
        self.cutoff = cutoff

    def save(self, file_dir):
        try:
            os.makedirs(file_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file_path = os.path.join(file_dir, 'model_param.json')
        params = {
            'type': 'powerlaw_hawkes',
            'params': {
                'names': ['baseline', 'multiplier', 'exponent', 'cutoff'],
                'values': [self.baseline, self.multiplier, self.exponent, self.cutoff],
            },
        }
        with open(file_path, 'w') as f:
            json.dump(params, f)

    def load(self, file_dir):
        file_path = os.path.join(file_dir, 'model_param.json')

        with open(file_path, 'r') as f:
            params = json.load(f)

        self.baseline, self.multiplier, self.exponent, self.cutoff = params['params']['values']

    def intensity(self, events):
        events = torch.Tensor(events)
        intensity = torch.zeros(len(events))
        for idx in range(len(events)):
            cur_t = events[idx]
            cur_hist = events[:idx]

            cur_int = self.baseline + torch.pow((self.cutoff + cur_t - cur_hist), -self.exponent - 1).sum() * self.multiplier
            intensity[idx] = cur_int

        return intensity

    def intensity_func(self, tau, hist, params):
        hist = hist.tolist()
        cur_time = tau + hist[-1] if hist else tau
        int_val = self.intensity(hist + [cur_time])[-1]
        return int_val

    def get_params(self):
        return self.baseline, self.multiplier, self.exponent, self.cutoff

    def upperbound(self, from_t, to_t, history):
        events = torch.empty(len(history) + 1)
        events[:-1] = history
        events[-1] = from_t
        return self.intensity(events)[-1]

    def generate_sequences(self, num_seqs, num_events):
        return generate_hawkes(self, num_seqs, num_events)

    def ll(self, batch, params=None):
        if params is None:
            params = self.get_params()

        seqs = []
        for idx in range(len(batch)):
            m = batch.mask[idx]
            cur_seq = batch.int_events[idx, m]
            first_event = batch.first_events[idx].item()
            new_seq = [first_event] + (first_event + cur_seq.cumsum(0)).tolist()

            seqs.append(new_seq)

        data = ((seq, params) for seq in seqs)

        loglikelihood = []
        for seq, d in zip(seqs, data):
            ll = loglikelihood_seq(d)

            ll /= len(seq)
            loglikelihood.append(ll)

#        with ProcessPoolExecutor() as executor:
#            for seq, ll in zip(seqs, executor.map(loglikelihood_seq, data)):
#
#                ll /= len(seq)
#                loglikelihood.append(ll)

        return np.array(loglikelihood, dtype='float64')

    def jacobian(self, batch, params=None):
        if params is None:
            params = self.get_params()

        seqs = []
        for idx in range(len(batch)):
            m = batch.mask[idx]
            cur_seq = batch.int_events[idx, m]
            first_event = batch.first_events[idx].item()
            new_seq = [first_event] + (first_event + cur_seq.cumsum(0)).tolist()

            seqs.append(new_seq)

        data = ((seq, params) for seq in seqs)

        derivs = []
        with ProcessPoolExecutor() as executor:
            for seq, deriv in zip(seqs, executor.map(jacobian_seq, data)):

                deriv /= len(seq)
                derivs.append(deriv)

        return np.array(derivs, dtype='float64')

    def fit(self, batch, args):

        def loss(params):
            params = (*params, self.cutoff)
            return -np.mean(self.ll(batch, params=params))

        def jac(params):
            params = (*params, self.cutoff)
            return -np.mean(self.jacobian(batch, params=params), axis=0)

        bounds = [
            (0, None),
            (1e-9, None),
            (1e-9, None),
        ]

        opt_res = minimize(
            loss, args['x0'], jac=jac, method=args['method'],
            bounds=bounds, options=args['options'])

        self.baseline, self.multiplier, self.exponent = map(float, opt_res.x)

        return opt_res

if __name__ == "__main__":
    process = PowerlawHawkes(0.5, 0.2, 0.3, 0.5)

    raw_data = process.generate_sequences(30, 100)
    data = []
    for idx in range(raw_data.shape[0]):
        data.append(raw_data[idx])

    dataset = GeneralDataset(data)
    batch = collate_no_marks(dataset)

    fit_process = PowerlawHawkes(1.0, 1.0, 0.3, 0.5)
    args = {'x0': [0.6, 0.3, 0.4], 'method': 'L-BFGS-B', 'fix_cutoff': True, 'options': {'disp': True}, 'workers': 4}
    res = fit_process.fit(batch, args)
    print(res)

# %%
