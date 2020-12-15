import torch
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from models.traditional.sequence_generators import generate_self_correcting
from models.traditional.abstract_traditional import AbstractTraditional
from models.data import GeneralDataset, collate_no_marks

def loglikelihood_seq(data):
    seq, params = data
    mu, alpha = params

    seq = [0] + seq

    cur_ll = 0
    for i in range(len(seq) - 1):
        t = seq[i+1]

        cur_ll += t * mu - i * alpha
        cur_ll -= np.exp(-alpha * i) / mu * \
            (np.exp(seq[i+1] * mu) - np.exp(seq[i] * mu))

    return cur_ll

class SelfCorrecting(AbstractTraditional):

    def __init__(self, mu, alpha):  # Better names?
        self.mu = mu
        self.alpha = alpha

    def save(self, file_dir):
        pass

    def load(self, file_dir):
        pass

    def intensity(self, events):
        events = torch.Tensor(events)
        num_events = torch.arange(len(events))

        intensity = torch.exp(self.mu * events - self.alpha * num_events)
        return intensity

    def intensity_func(self, tau, hist, params):
        hist = hist.tolist()
        cur_time = tau + hist[-1] if hist else tau
        int_val = self.intensity(hist + [cur_time])[-1]
        return int_val

    def get_params(self):
        return self.mu, self.alpha

    def upperbound(self, from_t, to_t, history):
        return float('inf')  # For completeness

    def generate_sequences(self, num_seqs, num_events):
        return generate_self_correcting(self, num_seqs, num_events)

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