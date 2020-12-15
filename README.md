# Code appendix

Code appendix for "UNIPoint: Universally Approximating Point Processes Intensities".

## Install Requirements

Simply run
```
pip3 install .
```

## Datasets
[Download](https://drive.google.com/drive/folders/1zOUbnfYUJGNNdxYTZ_IB148btiNxHaX2?usp=sharing) and unzip files into the `data` folder.

## How to run

There are two sets of experiments available: (1) synthetic datasets; (2) real world datasets.

### Synthetic

First train the models

```
python3 train_synth.py
```

Then evaluate for either log-likelihood and/or total variation
```
python3 eval_ll_synth.py
python3 eval_tv_synth.py
```

Resulting files of evaluation metrics per test sequence can be found in the `eval` folder.

### Real world

Similar to synthetic, train the models first
```
python3 train_real.py
```

Then evaluate the log-likelihood
```
python3 eval_ll_real.py
```

The evaluated log-likelihood values are in the `eval` folder
