import torch
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from models.traditional.sequence_generators import generate_hawkes
from models.traditional.abstract_traditional import AbstractTraditional
from models.data import GeneralDataset, collate_no_marks

def loglikelihood_seq(data):
    seq, params = data
    mu, alpha, beta = params

    max_t = seq[-1]
    seq = seq[:-1]

    cur_ll = 0
    for i in range(len(seq)):
        t = seq[i]

        intensity = mu + alpha * np.sin(2 * np.pi * t / beta)
        cur_ll += np.log(intensity)

    cur_ll -= max_t * mu
    cur_ll -= alpha * beta / np.pi * np.sin(np.pi * max_t / beta) ** 2

    return cur_ll

class InhomogeneousSinPoisson(AbstractTraditional):

    def __init__(self, baseline, magnitude, frequency):
        self.baseline = baseline
        self.magnitude = magnitude
        self.frequency = frequency

    def save(self, file_dir):
        pass

    def load(self, file_dir):
        pass

    def intensity(self, events):
        events = torch.Tensor(events)
        intensity = self.magnitude * torch.sin(2 * np.pi * events / self.frequency) + self.baseline

        return intensity

    def intensity_func(self, tau, hist, params):
        hist = hist.tolist()
        cur_time = tau + hist[-1] if hist else tau

        baseline, magnitude, frequency = params
        return magnitude * np.sin(2 * np.pi * cur_time / frequency) + baseline

    def get_params(self):
        return self.baseline, self.magnitude, self.frequency

    def upperbound(self, from_t, to_t, history):
        return self.magnitude + self.baseline

    def generate_sequences(self, num_seqs, num_events):
        return generate_hawkes(self, num_seqs, num_events)

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
