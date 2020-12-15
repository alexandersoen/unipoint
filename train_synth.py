#%%
ROOT_DATA_FOLDER = 'data/synth'
ROOT_SAVE_FOLDER = 'save'

#%%
# Currently just a test. Need to verify the sequence generation algorithm
import os
import csv
import torch
import itertools

from models.basis_sum.exp_basis import ExpSum
from models.basis_sum.powerlaw_basis import PowerlawSum
from models.basis_sum.relu_basis import ReLUSum
from models.basis_sum.sin_basis import SinSum
from models.basis_sum.sigmoid_basis import SigmoidSum
from models.basis_sum.mixed_basis import MixedSum

from models.data import GeneralDataset, collate_no_marks

#%%
#RNN_SIZE = 32
RNN_SIZE = 48
RNN_LAYER = 1
NUM_BASIS = 64
NUM_OUTPUT_LAYER = 2
MC_SIMS = 1

BATCH_SIZE = 64
EPOCHS = 500
PATIENCE = 10
LR=1e-3
L2REG = 1e-5

#%%
synth_data_list = [
    'homogeneous',
    'inhomogeneous',
    'selfcorrecting',
    'exphawkes',
    'powerlawhawkes',
    'sinexphawkes',
]

make_model = [
    ('expsum', lambda x: ExpSum(RNN_SIZE, RNN_LAYER, NUM_BASIS, MC_SIMS)),
    ('powerlaw', lambda x: PowerlawSum(RNN_SIZE, RNN_LAYER, NUM_BASIS, MC_SIMS)),
    ('relusum', lambda x: ReLUSum(RNN_SIZE, RNN_LAYER, NUM_BASIS, MC_SIMS)),
    ('sinsum', lambda x: SinSum(RNN_SIZE, RNN_LAYER, NUM_BASIS, MC_SIMS)),
    ('sigmoidsum', lambda x: SigmoidSum(RNN_SIZE, RNN_LAYER, NUM_BASIS, MC_SIMS)),
    ('mixedsum', lambda x: MixedSum(RNN_SIZE, RNN_LAYER, NUM_BASIS // 2, MC_SIMS)),
]

#%%
for name, model_vals in itertools.product(synth_data_list, make_model):
    save_file = os.path.join(ROOT_DATA_FOLDER, name + '.csv')

    model_name, model_gen = model_vals
    model = model_gen(True)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('[Info] Number of parameters: {}'.format(num_params))

    save_name = name + '_' + model_name

    print('Start Processing:', save_name)

    data = []
    with open(save_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            data.append(torch.Tensor(list(map(float, row))))

    whole_dataset = GeneralDataset(data)
    whole_dataset.log_transform_rnn()

    train_dataset, validation_dataset, test_dataset = whole_dataset.train_val_test_split(0.6, 0.2, 0.2, seed=1)

    rnn_mean, rnn_std = train_dataset.rnn_statistics()

    train_dataset.normalise_data(rnn_mean=rnn_mean, rnn_std=rnn_std)
    validation_dataset.normalise_data(rnn_mean=rnn_mean, rnn_std=rnn_std)
    test_dataset.normalise_data(rnn_mean=rnn_mean, rnn_std=rnn_std)

    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=L2REG)
    save_path = os.path.join(ROOT_SAVE_FOLDER, save_name)
    valid_hist = model.train_model(
        train_dataset, EPOCHS, opt, batch_size=BATCH_SIZE,
        validation_data=validation_dataset, patience=PATIENCE, save_model=save_path, from_checkpoint=True)

    model.save(save_path)

    print('Finish Processing:', save_name)