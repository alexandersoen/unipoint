import torch

from abc import ABC, abstractmethod

class AbstractModel(ABC):

    @abstractmethod
    def save(self, file_dir):
        pass

    @abstractmethod
    def load(self, file_dir):
        pass

    @abstractmethod
    def intensity(self, events):
        pass

    @abstractmethod
    def intensity_func(self, tau, hist, params):
        pass

    @abstractmethod
    def event_params(self, batch):
        pass

    @abstractmethod
    def ll(self, batch):
        pass

    def aggregate(self, ll, batch):
        return torch.mean(ll)
#        return ll.sum() / sum(batch.seq_lengths)