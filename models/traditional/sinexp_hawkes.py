import torch
import numpy as np

from concurrent.futures import ProcessPoolExecutor

from models.traditional.sequence_generators import generate_hawkes
from models.traditional.abstract_traditional import AbstractTraditional
from models.data import GeneralDataset, collate_no_marks

def int_term(alpha, beta, s):
    return - np.exp(-beta * s) * (alpha ** 2 - beta ** 2 * np.sin(-alpha * s) + alpha * beta * np.cos(-alpha * s) + beta ** 2)

def loglikelihood_seq2(data):
    seq, params = data
    mu, alpha, beta, gamma = params
    alpha = 2 * np.pi / alpha

    max_t = seq[-1]
    seq = seq[:-1]

    prev_seq = np.zeros(len(seq)+1)
    prev_seq[1:] = seq

    cur_ll = 0
    comp = 0
    for i in range(len(seq)):
        prev_t = prev_seq[i]
        t = seq[i]
        history = seq[:i]

        kernel_vals = ((1 + np.sin(alpha*(t-h))) * np.exp(-beta*(t-h)) for h in history)
        intensity =  mu + gamma * sum(kernel_vals)

        cur_ll += np.log(intensity)

        samples = np.random.uniform(prev_t, t, 20)
        comp_val = 0
        for s in samples:
            kernel_vals = ((1 + np.sin(alpha*(s-h))) * np.exp(-beta*(s-h)) for h in history)
            intensity =  mu + gamma * sum(kernel_vals)

            comp_val += intensity

        comp_val *= (t - prev_t) / 20

        cur_ll -= comp_val

    return cur_ll


def loglikelihood_seq(data):
    seq, params = data
    mu, alpha, beta, gamma = params
    alpha = 2 * np.pi / alpha

    max_t = seq[-1]
    seq = seq[:-1]

    prev_seq = np.zeros(len(seq)+1)
    prev_seq[1:] = seq

    cur_ll = 0
    comp = 0
    for i in range(len(seq)):
        prev_t = prev_seq[i]
        t = seq[i]
        history = seq[:i]

        kernel_vals = ((1 + np.sin(alpha*(t-h))) * np.exp(-beta*(t-h)) for h in history)
        intensity =  mu + gamma * sum(kernel_vals)

        cur_ll += np.log(intensity)

        # Compensator terms
        for j in range(i):
            comp += int_term(alpha, beta, t - seq[j]) - int_term(alpha, beta, prev_t - seq[j])

    for j in range(len(seq)):
        comp += int_term(alpha, beta, max_t - seq[j]) - int_term(alpha, beta, seq[-1] - seq[j])

    comp *= gamma / (beta * (alpha ** 2 + beta ** 2))
    comp += max_t * mu

    cur_ll -= comp

    return cur_ll

class SinExpHawkes(AbstractTraditional):

    def __init__(self, baseline, scale, frequency, decay):
        self.baseline = baseline
        self.scale = scale
        self.frequency = frequency
        self.decay = decay

    def save(self, file_dir):
        pass

    def load(self, file_dir):
        pass

    def intensity(self, events):
        events = torch.Tensor(events)
        intensity = torch.zeros(len(events))
        for idx in range(len(events)):
            cur_t = events[idx]
            cur_hist = events[:idx]

            cur_int = self.baseline + ((1 + torch.sin(2 * np.pi * (cur_t - cur_hist) / self.frequency)) * torch.exp(- self.decay * (cur_t - cur_hist))).sum() * self.scale
            intensity[idx] = cur_int

        return intensity

    def intensity_func(self, tau, hist, params):
        hist = hist.tolist()
        cur_time = tau + hist[-1] if hist else tau
        int_val = self.intensity(hist + [cur_time])[-1]
        return int_val

    def get_params(self):
        return self.baseline, self.scale, self.frequency, self.decay

    def upperbound(self, from_t, to_t, history):
        ukernel = 2 * torch.exp(- self.decay * (from_t - history))
        return self.baseline + self.scale * ukernel.sum()

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