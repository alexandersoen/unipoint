#%%
ROOT_DATA_FOLDER = 'data/synth'
ROOT_SAVE_FOLDER = 'save'
ROOT_LL_FOLDER = 'eval/ll'

#%%
# Imports
import os
import csv
import torch
import numpy as np
import itertools

from models.traditional.self_correcting import SelfCorrecting
from models.traditional.exp_hawkes import ExpHawkes
from models.traditional.sinexp_hawkes import SinExpHawkes

from models.basis_sum.exp_basis import ExpSum
from models.basis_sum.powerlaw_basis import PowerlawSum
from models.basis_sum.relu_basis import ReLUSum
from models.basis_sum.sin_basis import SinSum
from models.basis_sum.sigmoid_basis import SigmoidSum
from models.basis_sum.mixed_basis import MixedSum

from models.data import GeneralDataset, collate_no_marks
from torch.utils.data import Dataloader

from plot.utils import total_variation

#%%
RNN_SIZE = 48
RNN_LAYER = 1
NUM_BASIS = 64
NUM_OUTPUT_LAYER = 2

synth_data_list = [
    ('selfcorrecting', lambda x: SelfCorrecting(1, 1)),
    ('exphawkes', lambda x: ExpHawkes(0.5, 0.8, 1)),
    ('sinexphawkes', lambda x: SinExpHawkes(0.5, 0.4, 2, 1.0)),
]

make_model = [
    ('expsum', lambda x: ExpSum(RNN_SIZE, RNN_LAYER, NUM_BASIS)),
    ('powerlaw', lambda x: PowerlawSum(RNN_SIZE, RNN_LAYER, NUM_BASIS)),
    ('relusum', lambda x: ReLUSum(RNN_SIZE, RNN_LAYER, NUM_BASIS)),
    ('sinsum', lambda x: SinSum(RNN_SIZE, RNN_LAYER, NUM_BASIS)),
    ('sigmoidsum', lambda x: SigmoidSum(RNN_SIZE, RNN_LAYER, NUM_BASIS)),
    ('mixedsum', lambda x: MixedSum(RNN_SIZE, RNN_LAYER, NUM_BASIS // 2)),
]

#%%
data_model_iter = itertools.product(synth_data_list, make_model)
for dataset_pair, model_pair in data_model_iter:
    model_name, model_gen = model_pair
    dataset_name, dataset_gen = dataset_pair
    fitted_model = model_gen(True)
    true_model = dataset_gen(True)

    fitted_name = dataset_name + '_' + model_name
    true_name = dataset_name + '_' + dataset_name

    print('Start Evaluating:', fitted_name)

    fitted_folder = os.path.join(ROOT_SAVE_FOLDER, fitted_name)
    fitted_model.load(fitted_folder)

    data = []
    save_file = os.path.join(ROOT_DATA_FOLDER, dataset_name + '.csv')
    with open(save_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(torch.Tensor(list(map(float, row))))

    # Preprocess
    whole_dataset = GeneralDataset(data)
    whole_dataset.log_transform_rnn()

    train_dataset, validation_dataset, test_dataset = whole_dataset.train_val_test_split(0.6, 0.2, 0.2, seed=1)

    rnn_mean, rnn_std = train_dataset.rnn_statistics()

    test_dataset.normalise_data(rnn_mean=rnn_mean, rnn_std=rnn_std)

    try:
        test_dataloader = DataLoader(
            test_dataset, batch_size=1, shuffle=False, collate_fn=collate_no_marks)
        lls = []
        for test_collate in test_dataloader:
            lls.append(fitted_model.ll(test_collate).tolist()[0])

    except:
        test_collate = collate_no_marks(test_dataset)

        lls = fitted_model.ll(test_collate).tolist()

    ll_save_file = os.path.join(ROOT_LL_FOLDER, fitted_name + '.csv')

    with open(ll_save_file, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(lls)

    print('Average LL:', np.mean(lls))

    print('Finish Evaluating:', fitted_name)
