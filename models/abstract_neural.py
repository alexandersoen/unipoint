import os
import errno
import json
import math
import random

import numpy as np

import torch
import torch.nn as nn

from copy import deepcopy

from abc import ABC, abstractmethod

from tqdm import tqdm

from datetime import datetime

from torch.nn.utils.rnn import (pack_padded_sequence, pad_packed_sequence)
from torch.utils.data import DataLoader

from models.data import GeneralDataset, collate_no_marks
from models.abstract_model import AbstractModel

class AbstractNeural(AbstractModel, nn.Module, ABC):

    def __init__(
            self, recurrent_output_size, recurrent_num_layers,
            recurrent_kwarg):

        super().__init__()

        # Generate recurrent unit
        self.recurrent = self.recurrent_builder(
            input_size=1,
            output_size=recurrent_output_size,
            num_layers=recurrent_num_layers,
            kwargs=recurrent_kwarg)

        self.recurrent_output_size = recurrent_output_size
        self.recurrent_num_layers = recurrent_num_layers

    def save(self, file_dir):
        try:
            os.makedirs(file_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
        model_path = os.path.join(file_dir, 'model_save.pkl')
        torch.save(self.state_dict(), model_path)
        self.save_parameters(file_dir)

    def load(self, file_dir):
        model_path = os.path.join(file_dir, 'model_save.pkl')
        self.load_state_dict(torch.load(model_path))
        self.load_parameters(file_dir)

    def get_hidden_state(self, batch):
        x = batch.rnn_events.unsqueeze(-1)

        self.hidden = self.init_hidden(len(batch))

        # Pack sequence
        x = pack_padded_sequence(x, batch.seq_lengths, batch_first=True)

        h, _ = self.recurrent(x, self.hidden)
        h, _ = pad_packed_sequence(h, batch_first=True)
        return h

    def train_model(
            self, train_data, epoch, optimiser, batch_size=64,
            validation_data=None, patience=50, save_model=None,
            from_checkpoint=False):

        steps = 0
        length = math.ceil(len(train_data) / batch_size)
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True,
            collate_fn=collate_no_marks)

        validation_dataloader = DataLoader(
            validation_data, batch_size=batch_size, shuffle=False,
            collate_fn=collate_no_marks)

        val_best_loss = np.inf
        break_count = 0
        best_model = None

        valid_list = []

        if from_checkpoint and save_model is not None:
            try:
                save_start = 'epoch'
                get_epoch_num = lambda x: int(x[1+len(save_start):])

                folders = []
                for folder in os.listdir(save_model):
                    if folder[:len(save_start)] == save_start:
                        folders.append(folder)

                checkpoint = max(folders, key=get_epoch_num)

                checkpoint_path = os.path.join(save_model, checkpoint)

                print('\n[{}] ----- Loading checkpoint {} -----'.format(datetime.now(), get_epoch_num(checkpoint)))
                self.load(checkpoint_path)

                start = get_epoch_num(checkpoint) + 1
            except FileNotFoundError:
                start = 1
        else:
            start = 1

        for e in range(start , epoch+1):
            print('\n[{}] ----- Starting Epoch {} -----'.format(datetime.now(), e))

            batch_num = 0
            self.train()
            for _, batch in tqdm(enumerate(train_dataloader), total=length):

                optimiser.zero_grad()

                # Forward propagate + loss calculation
                loss = self.ll(batch)
                loss.create_graph = True

                total_loss = -self.aggregate(loss, batch)

                # Backward propagate
                total_loss.backward()
                optimiser.step()

                batch_num += 1
                steps += 1

            self.eval()

            if validation_data is not None:

                val_total_loss = 0
                for validation_batch in validation_dataloader:
                    val_loss = self.ll(validation_batch)
                    val_total_loss -= self.aggregate(val_loss, validation_batch).item()

                valid_list.append(val_total_loss)

                if (val_best_loss - val_total_loss) < 1e-4:
                    break_count += 1

                    if val_total_loss < val_best_loss:
                        val_best_loss = val_total_loss
                        best_model = deepcopy(self.state_dict())

                else:
                    val_best_loss = val_total_loss
                    best_model = deepcopy(self.state_dict())

                    break_count = 0
                
                if break_count > patience:
                    print('[{}] ##---- Early Stopped')
                    break

            if save_model:
                model_path = os.path.join(save_model, 'epoch_{}'.format(e))

                try:
                    os.makedirs(model_path)
                except OSError as er:
                    if er.errno != errno.EEXIST:
                        raise

            self.save(model_path)

            print('[{}] ##--- Update: Batch {} in Epoch {}. Train Loss={:.4f}; Validation Loss={:.4f}'.format(
                datetime.now(), batch_num+1, e, total_loss.item(), val_total_loss))

            print('[{}] ----- Finished Epoch {} -----\n'.format(
                datetime.now(), e))

        # Save best
        if best_model is not None:
            self.load_state_dict(best_model)

        return valid_list

    def intensity(self, events):
        dataset = GeneralDataset([events])
        batch = collate_no_marks(dataset)

        return self.forward(batch)[0]

    def event_params(self, batch):
        h = self.get_hidden_state(batch)
        return [p.squeeze() for p in self.get_params(h)]

    @abstractmethod
    def save_parameters(self, file_dir):
        pass

    @abstractmethod
    def load_parameters(self, file_dir):
        pass

    @abstractmethod
    def recurrent_builder(
            self, input_size, output_size, num_layers, kwargs):
        pass

    @abstractmethod
    def init_hidden(self, batch_size):
        pass

    @abstractmethod
    def get_params(self, h):
        pass

    @ abstractmethod
    def forward(self, batch):
        pass

    @abstractmethod
    def intensity_func(self, tau, params):
        pass