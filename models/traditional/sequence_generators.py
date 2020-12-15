import torch
import numpy as np

#def generate_homogeneous(rate, max_t):
#    seq = []
#    t = 0
#    while True:
#        dt = np.random.exponential(1 / rate )
#        new_t = t + dt
#
#        if new_t > max_t:
#            break
#
#        seq.append(new_t)
#        t = new_t
#
#    return torch.Tensor(seq)

def generate_homogeneous(process, num_seqs, num_events):
    rate = process.rate
    dts = torch.empty((num_seqs, num_events)).exponential_(lambd=rate)

    return dts.cumsum(-1)

def generate_hawkes(process, num_seqs, num_events):  # Works with inhomogeneous
    seq_list = []
    for _ in range(num_seqs):
        new_seq = generate_hawkes_single(process, num_events)
        seq_list.append(new_seq)
    
    return torch.stack(seq_list)

def generate_hawkes_single(process, num_events):

    count = 0
    cur_t = 0
    seq = []
    while True:
        upper = process.upperbound(cur_t, float('inf'), torch.Tensor(seq))
        dt = np.random.exponential(1 / upper)

        new_t = cur_t + dt
        new_int = process.intensity(seq + [new_t])[-1]

        # Accept/Reject
        uniform = np.random.uniform()
        if uniform <= new_int / upper:
            seq.append(new_t)
            count += 1

            if count == num_events:
                break

        cur_t = new_t

    return torch.Tensor(seq)

def generate_self_correcting(process, num_seqs, num_events):
    mu, alpha = process.get_params()

    cur_t = torch.zeros(num_seqs)

    dts = torch.empty((num_seqs, num_events))
    for idx in range(num_events):
        e_vec = torch.empty(num_seqs).exponential_()
        next_dt = torch.log(e_vec * mu / torch.exp(cur_t) + 1) / mu

        dts[:, idx] = next_dt
        cur_t += mu * next_dt - alpha

    return dts.cumsum(-1)