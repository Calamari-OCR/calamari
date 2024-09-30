
![logo](resources/logo/calamari_200.png)

[![Python Test](https://github.com/Calamari-OCR/calamari/actions/workflows/python-test.yml/badge.svg)](https://github.com/Calamari-OCR/calamari/actions/workflows/python-test.yml)
[![codecov](https://codecov.io/github/bertsky/calamari/graph/badge.svg?token=JKGPINQTKO)](https://codecov.io/github/bertsky/calamari)
[![Lint](https://github.com/Calamari-OCR/calamari/actions/workflows/black.yml/badge.svg)](https://github.com/Calamari-OCR/calamari/actions/workflows/black.yml)
[![Upload Python Package](https://github.com/Calamari-OCR/calamari/actions/workflows/python-publish.yml/badge.svg)](https://github.com/Calamari-OCR/calamari/actions/workflows/python-publish.yml)
[![image](https://img.shields.io/pypi/v/calamari_ocr.svg)](https://pypi.org/project/calamari_ocr/)


OCR Engine based on OCRopy and Kraken using Python 3.

It is designed to both be easy to use from the command line but also be modular to be integrated and customized from other python scripts.

![preview](resources/preview.png)

## Documentation

The documentation of Calamari is hosted [here](https://calamari-ocr.readthedocs.io).

##  Pretrained model repository

Pretrained models are available at [calamari_models](https://github.com/Calamari-OCR/calamari_models)
and [calamari_models_experimental](https://github.com/Calamari-OCR/calamari_models_experimental).

Current releases (with individual model tarballs) can be accessed
[here](https://github.com/Calamari-OCR/calamari_models/releases/tag/2.1) and
[here](https://github.com/Calamari-OCR/calamari_models_experimental/releases/tag/v0.0.3).

## Installing

Calamari is available on [pypi](https://pypi.org/project/calamari-ocr):

```shell
pip install calamari-ocr
```

Read the [docs](https://calamari-ocr.readthedocs.io) for further instructions.

## Command-Line Interface

See the [docs](https://calamari-ocr.readthedocs.io) to learn how to use Calamari from the command line.

## Calamari API

See the [docs](https://calamari-ocr.readthedocs.io) to learn how to adapt Calamari for your needs.

## Citing Calamari

If you use Calamari in your Research-Project, please cite:

> Wick, C., Reul, C., Puppe, F.: *Calamari - A High-Performance Tensorflow-based Deep Learning Package for Optical Character Recognition.* Digital Humanities Quarterly **14**(1) (2020)

```
@article{wick_calamari_2020,
    title = {Calamari - {A} {High}-{Performance} {Tensorflow}-based {Deep} {Learning} {Package} for {Optical} {Character} {Recognition}},
    volume = {14},
    number = {1},
    journal = {Digital Humanities Quarterly},
    author = {Wick, Christoph and Reul, Christian and Puppe, Frank},
    year = {2020},
}
```
