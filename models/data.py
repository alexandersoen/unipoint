import copy
import torch
import numpy as np

from sklearn.model_selection import train_test_split

from torch.utils.data import Dataset

from torch.nn.utils.rnn import pad_sequence

class GeneralDataset(Dataset):

    def __init__(self, time_data, mark_data=None, max_time=None, mark_domain=1):
        # Initialisation doesn't have to be super quick as only run once per training

        # Assumes time data starts at 0 with start mark value
        # Assumes inputs are in list(torch.Tensor) format
        # Assumes mark_domain includes (non-start) mark size

        # Generate the intensity and rnn event/marks seqs
        self.first_events = []
        self.int_events, self.rnn_events = [], []
        self.int_marks, self.rnn_marks = [], []

        if mark_data is None:
            # As seq contains 0th time marks of interarrival are same dimension
            mark_data = [torch.zeros_like(seq) for seq in time_data]
            for idx in range(len(mark_data)):
                # Separate starting token and events
                mark_data[idx][1:] += 1

        if max_time is not None:
            for idx in range(len(time_data)):
                # Add space for max time
                new_time_seq = torch.zeros(len(time_data[idx])+1)
                new_mark_seq = torch.zeros(len(mark_data[idx])+1)

                new_time_seq[:-1] = time_data[idx]
                new_mark_seq[:-1] = mark_data[idx]

                # Add max time values
                new_time_seq[-1] = max_time
                new_mark_seq[-1] = mark_domain + 1  # Finish mark

                time_data[idx] = new_time_seq
                mark_data[idx] = new_mark_seq

        # Turn into interarrival
        for idx in range(len(time_data)):
            self.first_events.append(time_data[idx][0])

            cur_time_seq = time_data[idx]
            cur_mark_seq = mark_data[idx]

            cur_inter_seq = np.concatenate([[1], np.ediff1d(cur_time_seq)])
            cur_inter_seq = torch.Tensor(cur_inter_seq)

            self.int_events.append(cur_inter_seq[1:])
            self.rnn_events.append(cur_inter_seq[:-1])

            self.int_marks.append(cur_mark_seq[1:])
            self.rnn_marks.append(cur_mark_seq[:-1])

        self.indices = torch.arange(len(self))
        self.split = False

    def log_transform_rnn(self):
        new_rnn_events = []
        for idx in range(len(self)):
            new_rnn_event = self.rnn_events[idx].clone()
            new_rnn_event[1:] = torch.log(new_rnn_event[1:] + 1e-8)

            new_rnn_events.append(new_rnn_event)
        self.rnn_events = new_rnn_events

        return self

    def normalise_data(self, rnn_mean=None, rnn_std=None):
        # RNN: Normalise by mean and std for
        if rnn_mean is None or rnn_std is None:
            rnn_mean, rnn_std = self.rnn_statistics()

        self.rnn_events = [(seq - rnn_mean) / rnn_std for seq in self.rnn_events]

        # Int: Normalise by std
        #int_events = torch.cat(self.int_events)
        #int_std = int_events.std()

        #self.int_events = [seq / int_std for seq in self.int_events]
        return self

    def sequence_length_split(self, split_length):

        new_first_events = []
        new_int_events, new_rnn_events = [], []
        new_int_marks, new_rnn_marks = [], []
        new_indices = []
        for idx in range(len(self)):
            cur_first_event = self.first_events[idx]
            cur_int_events = self.int_events[idx]
            cur_rnn_events = self.rnn_events[idx]
            cur_int_marks = self.int_marks[idx]
            cur_rnn_marks = self.rnn_marks[idx]

            num_batches = int(np.ceil(len(cur_int_events) / split_length))
            for i in range(num_batches):
                new_int_event = cur_int_events[i * split_length : (i+1) * split_length]
                new_rnn_event = cur_rnn_events[i * split_length : (i+1) * split_length]
                new_int_mark = cur_int_marks[i * split_length : (i+1) * split_length]
                new_rnn_mark = cur_rnn_marks[i * split_length : (i+1) * split_length]

                new_first_events.append(cur_first_event)
                new_int_events.append(new_int_event)
                new_rnn_events.append(new_rnn_event)
                new_int_marks.append(new_int_mark)
                new_rnn_marks.append(new_rnn_mark)

                new_indices.append(idx)

        self.first_events = new_first_events
        self.int_events = new_int_events
        self.rnn_events = new_rnn_events
        self.int_marks = new_int_marks
        self.rnn_marks= new_rnn_marks
        self.indices = torch.Tensor(new_indices)

        return self

    def train_val_test_split(self, train_p=0.6, val_p=0.2, test_p=0.2, seed=1):
        if self.split:
            raise ValueError('Already split')
        # Split the dataloader into 3 more
        torch.manual_seed(seed)
        np.random.seed(seed)

        train_index, other_index = train_test_split(
            np.array(self.indices), train_size=train_p, test_size=(val_p + test_p))

        if val_p == 0:
            val_index = []
            test_index = other_index
        else:
            new_val_p = val_p / (val_p + test_p)
            new_test_p = 1 - new_val_p

            test_index, val_index = train_test_split(
                other_index, train_size=new_val_p, test_size=new_test_p)

        def generate_split(indices):
#            if len(indices) < 1:
#                return None

            # Copy and subset wrt indices
            new = copy.deepcopy(self)

            new.first_events = [new.first_events[i] for i in indices]

            new.int_events = [new.int_events[i] for i in indices]
            new.rnn_events = [new.rnn_events[i] for i in indices]

            new.int_marks = [new.int_marks[i] for i in indices]
            new.rnn_marks = [new.rnn_marks[i] for i in indices]

            new.indices = torch.tensor(indices)
            new.split = True

            return new

        train_dataset = generate_split(train_index)
        val_dataset = generate_split(val_index)
        test_dataset = generate_split(test_index)

        return train_dataset, val_dataset, test_dataset

    def rnn_statistics(self):
        stat_times = []

        for idx in range(len(self)):
            stat_times.append(self.rnn_events[idx][1:])

        stat_times = torch.cat(stat_times)

        return stat_times.mean(), stat_times.std()

    def __len__(self):
        return len(self.int_events)

    def __getitem__(self, index):
        return self.int_events[index], self.rnn_events[index],self.int_marks[index], self.rnn_marks[index], self.indices[index], self.first_events[index]

class Batch:
    def __init__(
            self, int_events, rnn_events, int_marks, rnn_marks, seq_lengths, indices, first_events):
        
        self.int_events, self.rnn_events = int_events, rnn_events
        self.int_marks, self.rnn_marks = int_marks, rnn_marks
        self.seq_lengths = seq_lengths
        self.indices = indices
        self.first_events = first_events

        m_row = torch.arange(len(rnn_events[0]))
        self.mask = m_row.repeat(rnn_events.shape[0], 1) < seq_lengths[:, None]

    def mc_values(self, sims):
        mc_events = self.int_events.repeat_interleave(sims, dim=-1)
        mc_mask = self.mask.repeat_interleave(sims, dim=-1)

        return mc_events, mc_mask

    def __len__(self):
        return self.int_events.shape[0]

def collate_no_marks(batch):

    batch = sorted(batch, key=lambda x: len(x[0]), reverse=True)

    seq_lengths = torch.tensor([len(seq[0]) for seq in batch])
    indices = torch.tensor([seq[4] for seq in batch])
    first_events = torch.tensor([seq[5] for seq in batch])

    int_events = pad_sequence([seq[0] for seq in batch], batch_first=True)
    rnn_events = pad_sequence([seq[1] for seq in batch], batch_first=True)

    int_marks = pad_sequence([seq[2] for seq in batch], batch_first=True)
    rnn_marks = pad_sequence([seq[3] for seq in batch], batch_first=True)

    rnn_events[:, 0] = 0

    return Batch(
        int_events, rnn_events, int_marks, rnn_marks, seq_lengths, indices, first_events)