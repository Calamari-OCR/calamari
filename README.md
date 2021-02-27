
![logo](resources/logo/calamari_200.png)

OCR Engine based on OCRopy and Kraken using python3.
It is designed to both be easy to use from the command line but also be modular to be integrated and customized from other python scripts.

![preview](resources/preview.png)

##  Pretrained model repository
Pretrained models are available at (https://github.com/Calamari-OCR/calamari_models).
The current release can be accessed [here](https://github.com/Calamari-OCR/calamari_models/archive/1.0.zip) (336 MB).

## Installing
### Installation using Pip
The suggested method is to install calamari into a virtual environment using pip:
```bash
virtualenv -p python3 PATH_TO_VENV_DIR # (e.g. virtualenv -p python3 calamari_venv)
source PATH_TO_VENV_DIR/bin/activate
pip install calamari_ocr
```
which will install Calamari and all of its dependencies.

To install the package without a virtual environment simply run
```
pip install calamari_ocr
```

To install the package from its source, download the source code and run
```
python setup.py install
```

### Installation using Conda
Run
```
conda env create -f environment_master.yml
```

## Command line interface (Standard User)
If you simply want to use calamari for applying existent models to your text lines and optionally train new models you probably should use the command line interface of calamari, which is very similar to the one of OCRopy.

Note that you have to `activate` the virtual environment if used during the installation in order to make the command line scripts available.

### Prediction of a page
Currently, only OCR on lines is supported.
To segment pages into lines (and the preceding preprocessing steps) we refer to the solutions provided by OCRopus, Kraken, Tesseract, etc.
For users (especially less technical ones) in need of an all-in-one package [OCR4all](https://github.com/OCR4all) might be worth a look.

The prediction step using very deep neural networks implemented on Tensorflow as core feature of calamari should be used:
```
calamari-predict --checkpoint path_to_model.ckpt --data.images your_images.*.png
```
Calamari also supports confidence-**voting** to improve different predictions of different models.
To enable voting you simply have to pass several models to the `--checkpoint` argument:
```
calamari-predict --checkpoint path_to_model_1.ckpt path_to_model_2.ckpt ... --data.images your_images.*.png
```

### Training of a model
In Calamari, you can both train a single model using a given data set or train a fold of several (default 5) models to generate different voters for a voted prediction.

#### Training a single model
A single model can be trained by the `calamar-train`-script. Given a data set with its ground truth you can train the default model by calling:
```
calamari-train --data.images your_images.*.png
```
Note, that calamari expects that each image file (.png) has a corresponding ground truth text file (.gt.txt) at the same location with the same base name.

There are several important parameters to adjust the training. For a full list type `calamari-train --help`.

#### Training-Params
- `--trainer.output_dir`: A path where to store checkpoints
- `--trainer.epochs`: The maximum number of training iterations (batches) for training. Note: this is the upper boundary if you use early stopping.
- `--trainer.samples_per_epoch`: The number of samples to process per epoch (by default the size of the dataset)
- `--early_stopping.frequency=1`: How often to check for early stopping on the validation dataset.
- `--early_stopping.n_to_go=10`: How many successive models must be worse than the current best model to break the training loop
- `--warmstart.model`: Load network weights from a given pretrained model. Note that the codec will probably change its size to match the codec of the provided ground truth files. To enforce that some characters may not be deleted use a `--whitelist`.
- `--n_augmentations=0`: Data augmentation on the training set.

#### Data-Params
- `--data.line_height 48`: The height of each rescaled input file passed to the network.
- `--val.images None`: Validation Images
- `--train.num_threads 1`: The number of threads used during training and line preprocessing.
- `--train.batch_size 1`: The number of lines processed in parallel.
- `--val.num_threads 1`: The number of threads used during training and line preprocessing (for validation).
- `--val.batch_size 1`: The number of lines processed in parallel (for validation).
- `--codec.whitelist` or `--codec.whitelist_files`: Specify either individual characters, or a text file listing all white list characters stored as string.

#### Model-Params
- `--network cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5`: Specify the network structure in a simple language. The default network consists of a stack of two CNN- and Pooling-Layers, respectively and a following LSTM layer. The network uses the default CTC-Loss implemented in Tensorflow for training and a dropout-rate of 0.5. The creation string thereto is: `cnn=40:3x3,pool=2x2,cnn=60:3x3,pool=2x2,lstm=200,dropout=0.5`. To add additional layers or remove a single layer just add or remove it in the comma separated list. Note that the order is important!

Hint: If you want to use early stopping but don't have a separated validation set you can train a single fold of the `calamari-cross-fold-train`-script (see next section).

### Training a n-fold of models
To train n more-or-less individual models given a training set you can use the `calamari-cross-fold-train`-script. The default call is
```
calamari-cross-fold-train --data.images your_images*.*.png --best_models_dir some_dir
```
By default, this will train 5 default models using 80%=(n-1)/n of the provided data for training and 20%=1/n for validation. These independent models can then be used to predict lines using a voting mechanism.
There are several important parameters to adjust the training. For a full list type `calamari-cross-fold-train --help`.

 - Almost parameters of `calamari-train` can be used to affect the training
 - `--n_folds=5`: The number of folds
 - `--weights=None`: Specify one or `n_folds` models to use for pretraining.
 - `--best_models_dir=REQUIRED`: Directory where to store the best model determined on the validation data set
 - `--best_model_label={id}`: The prefix for each of the best model of each fold. A string that will be formatted. `{id}` will be replaced by the number of the fold, i.e. 0, ..., n-1.
 - `--temporary_dir=None`: A directory where to store temporary files, e.g. checkpoints of the scripts to train an individual model. By default, a temporary dir using pythons `tempfile` modules is used.
 - `--max_parallel_models=n_folds`: The number of models that shall be run in parallel. By default, all models are trained in parallel.
 - `--single_fold=[]`: Use this parameter to train only a subset, e.g. a single fold out of all `n_folds`.
 
 To use all models to predict and then vote for a set of lines you can use the `calamari-predict` script and provide all models as `checkpoint`:
```
calamari-predict --checkpoint best_models_dir/*.ckpt.json --data.images your_images.*.png
```

### Evaluating a model
To compute the performance of a model you need first to predict your evaluation data set (see `calamari-predict`). Afterwards run
```
calamari-eval --gt.texts *.gt.txt
```
on the ground truth files to compute an evaluation measure including the full confusion matrix. By default, the predicted sentences as produced by the `calamari-predict` script end in `.pred.txt`. You can change the default behavior of the validation script by the following parameters

 - `--gt.texts=REQUIRED`: The ground truth txt files.
 - `--gt.pred_extension=.pred.txt`: The suffix of the prediction files if `--pred` is not specified
 - `--n_confusions=-1`: Print only the top `n_confusions` most common errors.
