import os
import csv
import json
import errno
import signal
import torch

from abc import ABC, abstractmethod

from models.abstract_model import AbstractModel

def handler(signum, fram):
    raise TimeoutError

class AbstractTraditional(AbstractModel, ABC):

    @abstractmethod
    def save(self, file_dir):  # TODO DEFINE
        pass

    @abstractmethod
    def load(self, file_dir):  # TODO DEFINE
        pass

    @abstractmethod
    def intensity(self, events):
        pass

    @abstractmethod
    def intensity_func(self, tau, params):
        pass

    @abstractmethod
    def get_params(self):
        pass

    @abstractmethod
    def upperbound(self, from_t, to_t, history):
        pass

    @abstractmethod
    def generate_sequences(self, num_seqs, num_events):  # Maybe switch to number of events?  Also make it torch
        pass

    def event_params(self, batch):  # Assume batch only has a single sequence
        params = self.get_params()
        e_params = torch.zeros(len(batch.int_events[0]), len(params))

        for idx, p in enumerate(params):
            e_params[:, idx] = p

        return e_params.split(1, dim=-1)