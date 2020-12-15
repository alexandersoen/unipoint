#%%
import os
import json
import errno
import torch
import numpy as np

from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

from models.traditional.sequence_generators import generate_homogeneous
from models.traditional.abstract_traditional import AbstractTraditional
from models.data import GeneralDataset, collate_no_marks

def loglikelihood_seq(data):
    seq, params = data
    rate = params[0]

    max_t = seq[-1]
    seq = seq[:-1]

    cur_ll = 0

    cur_ll += len(seq) * np.log(rate)
    cur_ll -= rate * max_t

    return cur_ll

def jacobian_seq(data):
    seq, params = data
    rate = params[0]

    max_t = seq[-1]
    seq = seq[:-1]

    return np.array([len(seq) / rate - max_t])

class HomogeneousPoisson(AbstractTraditional):

    def __init__(self, rate):
        self.rate = rate

    def save(self, file_dir):
        try:
            os.makedirs(file_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

        file_path = os.path.join(file_dir, 'model_param.json')
        params = {
            'type': 'homogeneous_poisson',
            'params': {
                'names': ['rate'],
                'values': [self.rate],
            },
        }
        with open(file_path, 'w') as f:
            json.dump(params, f)

    def load(self, file_dir):
        file_path = os.path.join(file_dir, 'model_param.json')

        with open(file_path, 'r') as f:
            params = json.load(f)

        (self.rate, ) = params['params']['values']

    def intensity(self, events):
        return torch.zeros(len(events)) + self.rate

    def intensity_func(self, tau, hist, params):
        rate = params

        return rate

    def get_params(self):
        return (self.rate, )

    def upperbound(self, from_t, to_t, history):
        return self.rate

    def generate_sequences(self, num_seqs, num_events):
        return generate_homogeneous(self, num_seqs, num_events)

    def ll(self, batch, params=None):
        if params is None:
            params = self.get_params()

        seqs = []
        for idx in range(len(batch)):
            m = batch.mask[idx]
            cur_seq = batch.int_events[idx, m]
            first_event = batch.first_events[idx]
            new_seq = [first_event] + (first_event + cur_seq.cumsum(0)).tolist()

            seqs.append(new_seq)

        data = ((seq, params) for seq in seqs)

        loglikelihood = []
        with ProcessPoolExecutor() as executor:
            for seq, ll in zip(seqs, executor.map(loglikelihood_seq, data)):

                ll /= len(seq)
                loglikelihood.append(ll)

        return np.array(loglikelihood, dtype='float64')

    def jacobian(self, batch, params=None):
        if params is None:
            params = self.get_params()

        seqs = []
        for idx in range(len(batch)):
            m = batch.mask[idx]
            cur_seq = batch.int_events[idx, m]
            first_event = batch.first_events[idx]
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
            (1e-9, None),
        ]

        opt_res = minimize(
            loss, args['x0'], jac=jac, method=args['method'],
            bounds=bounds, options=args['options'])

        self.rate = list(map(float, opt_res.x))[0]

        return opt_res

if __name__ == "__main__":
    process = HomogeneousPoisson(5)

    raw_data = process.generate_sequences(30, 100)
    data = []
    for idx in range(raw_data.shape[0]):
        data.append(raw_data[idx])

    dataset = GeneralDataset(data)
    batch = collate_no_marks(dataset)

    fit_process = HomogeneousPoisson(1)
    args = {'x0': [10], 'method': 'L-BFGS-B', 'fix_cutoff': True, 'options': {'disp': True}, 'workers': 4}
    res = fit_process.fit(batch, args)
    print(res)

# %%
