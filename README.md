# calamari
OCR Engine based on OCRopy and Kraken.
It is designed to both be easy to use from the command line but also be modular to be integrated and customized from other python scripts.

## Installing
The suggested method is to install calamari into a virtual environment using pip:
```
virtualenv PATH_TO_VENV_DIR (e. g. virtualenv calamari_venv)
source PATH_TO_VENV_DIR/bin/activate
pip install calamari
```
which will install calamari and all of its dependencies including tensorflow as default backend.

To install the package without a virtual environment simply run
```
pip install calamari
```

To install the package from its source, download the source code and run
```
python setup,py install
```

## Command line interface (Standard User)
If you simply want to use calamari for applying existent models to your text lines and optionally train new models you probably should use the command line interface of calamari, which is very similar to the one of OCRopy.

Note that you have to `activate` the virtual environment if used during the installation in order to make the command line skripts available.

### Prediction of a page
Currently only OCR on lines is supported.
Modules to segment pages into lines will be available soon.
In the meantime you should use the scripts provided by OCRopus.

The prediction step using very deep neural networks implemented on tensorflow as core feature of calamari should be used:
```
calamari-predict --checkpoint path_to_model.ckpt your_images.*.png
```
