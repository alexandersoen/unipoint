#%%
import os
import json
import torch
import errno
import numpy as np

from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

from models.traditional.sequence_generators import generate_hawkes
from models.traditional.abstract_traditional import AbstractTraditional
from models.data import GeneralDataset, collate_no_marks

def loglikelihood_seq(data):
    seq, params = data
    baseline, magnitude, decay = params

    max_t = seq[-1]
    seq = seq[:-1]

    cur_ll = 0
    for i in range(len(seq)):
        t = seq[i]
        history = seq[:i]

        kernel_vals = (np.exp(-decay * (t - h)) for h in history)
        intensity = baseline + magnitude * decay * sum(kernel_vals)

        cur_ll += np.log(1e-8 + intensity)

    cur_ll -= baseline * max_t
    cur_ll += magnitude * sum(np.exp(-decay * (max_t - t)) for t in seq)
    cur_ll -= magnitude * len(seq)

    return cur_ll

def jacobian_seq(data):
    seq, params = data
    mu, alpha, beta = params

    max_t = seq[-1]
    seq = seq[:-1]

    cur_mu_deriv = 0
    cur_alpha_deriv = 0
    cur_beta_deriv = 0

    for i in range(len(seq)):
        t = seq[i]
        history = seq[:i]

        kernel_vals = (np.exp(-beta * (t - h)) for h in history)
        intensity = mu + alpha * beta * sum(kernel_vals)

        cur_mu_deriv += 1 / (intensity + 1e-9)

        cur_alpha_deriv += beta * \
            sum(np.exp(-beta * (t - h)) / intensity for h in history)
        cur_alpha_deriv += np.exp(-beta * (max_t - t))

        cur_beta_deriv += alpha * \
            sum(np.exp(-beta * (t - h)) * (1 - beta * (t - h)) / intensity for h in history)
        cur_beta_deriv -= alpha * (max_t - t) * np.exp(-beta * (max_t - t))

    cur_mu_deriv -= max_t
    cur_alpha_deriv -= len(seq)

    return np.array([cur_mu_deriv, cur_alpha_deriv, cur_beta_deriv])

class ExpHawkes(AbstractTraditional):

    def __init__(self, baseline, magnitude, decay):
        self.baseline = baseline
        self.magnitude = magnitude
        self.decay = decay

    def save(self, file_dir):
        try:
            os.makedirs(file_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file_path = os.path.join(file_dir, 'model_param.json')
        params = {
            'type': 'exp_hawkes',
            'params': {
                'names': ['baseline', 'magnitude', 'decay'],
                'values': [self.baseline, self.magnitude, self.decay],
            },
        }
        with open(file_path, 'w') as f:
            json.dump(params, f)

    def load(self, file_dir):
        file_path = os.path.join(file_dir, 'model_param.json')

        with open(file_path, 'r') as f:
            params = json.load(f)

        self.baseline, self.magnitude, self.decay = params['params']['values']

    def intensity(self, events):
        events = torch.Tensor(events)
        intensity = torch.zeros(len(events))
        for idx in range(len(events)):
            cur_t = events[idx]
            cur_hist = events[:idx]

            cur_int = self.baseline + (self.magnitude * self.decay * torch.exp(-self.decay * (cur_t - cur_hist))).sum()
            intensity[idx] = cur_int

        return intensity

    def intensity_func(self, tau, hist, params):
        if type(hist) != list:
            hist = hist.tolist()

        cur_time = tau + hist[-1] if hist else tau
        int_val = self.intensity(hist + [cur_time])[-1]
        return int_val

    def get_params(self):
        return self.baseline, self.magnitude, self.decay

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
#        for seq, d in zip(seqs, data):
#            ll = loglikelihood_seq(d)
#
#            ll /= len(seq)
#            loglikelihood.append(ll)
#        ll_val = 0
        with ProcessPoolExecutor() as executor:
            for seq, ll in zip(seqs, executor.map(loglikelihood_seq, data)):

                ll /= len(seq)
#                ll_val += ll
                loglikelihood.append(ll)

#        return np.array([ll_val / len(seqs)], dtype='float64')
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
            return -np.mean(self.ll(batch, params=params))

        def jac(params):
            return -np.mean(self.jacobian(batch, params=params), axis=0)

        bounds = [
            (0, None),
            (0, None),
            (0, None),
        ]

        opt_res = minimize(
            loss, args['x0'], jac=jac, method=args['method'],
            bounds=bounds, options=args['options'])

        self.baseline, self.magnitude, self.decay = map(float, opt_res.x)

        return opt_res

if __name__ == "__main__":
    process = ExpHawkes(0.5, 0.8, 1.0)

    raw_data = process.generate_sequences(30, 100)
    data = []
    for idx in range(raw_data.shape[0]):
        data.append(raw_data[idx])

    dataset = GeneralDataset(data)
    batch = collate_no_marks(dataset)

    fit_process = ExpHawkes(2, 2, 2)
    args = {'x0': [0.3, 0.3, 0.5], 'method': 'L-BFGS-B', 'options': {'disp': True, 
    'ftol': 1e-05, 'gtol': 1e-04}}
    res = fit_process.fit(batch, args)
    print(res)

# %%
