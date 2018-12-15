from .dataset import DataSet, DataSetMode, RawDataSet
from calamari_ocr.ocr.data_processing import DataPreprocessor
from calamari_ocr.ocr.text_processing import TextProcessor
from calamari_ocr.ocr.augmentation import DataAugmenter
from typing import Generator, Tuple, List, Any
import numpy as np


def RawInputDataset(
        mode: DataSetMode,
        raw_datas, raw_texts, raw_params,
        data_preprocessor, text_preprocessor,
        data_augmenter=None, data_augmentation_amount=0
):
    dataset = InputDataset(RawDataSet(mode=mode, images=raw_datas, texts=raw_texts),
                           data_preprocessor,
                           text_preprocessor,
                           data_augmenter, data_augmentation_amount
                           )
    dataset.preloaded_datas = raw_datas
    dataset.preloaded_texts = raw_texts
    dataset.preloaded_params = raw_params
    return dataset


class InputDataset:
    def __init__(self,
                 dataset: DataSet,
                 data_preprocessor: DataPreprocessor,
                 text_preprocessor: TextProcessor,
                 data_augmenter: DataAugmenter = None,
                 data_augmentation_amount: float = 0,
                 skip_invalid_gt=True):
        self.dataset = dataset
        self.data_processor = data_preprocessor
        self.text_processor = text_preprocessor
        self.skip_invalid_gt = skip_invalid_gt
        self.data_augmenter = data_augmenter
        self.preloaded_datas = []
        self.preloaded_texts = []
        self.preloaded_params = []
        self.data_augmentation_amount = data_augmentation_amount
        self.generate_only_non_augmented = False

        if data_augmenter and dataset.mode != DataSetMode.TRAIN:
            raise Exception('Data augmentation is only supported for training, but got {} dataset instead'.format(dataset.mode))

        if data_augmentation_amount > 0 and self.data_augmenter is None:
            raise Exception('Requested data augmentation, but no data augmented provided. Use e. g. SimpleDataAugmenter')

    def __len__(self):
        return len(self.dataset.samples())

    def preload(self, processes=1, progress_bar=False):
        print("Preloading dataset type {} with size {}".format(self.dataset.mode, len(self)))
        self.dataset.load_samples(processes=1, progress_bar=progress_bar)           # load data always with one thread
        datas, txts = self.dataset.train_samples(skip_empty=self.skip_invalid_gt)

        if self.text_processor:
            texts = self.text_processor.apply(txts, processes=processes, progress_bar=progress_bar)
        if self.data_processor:
            datas, params = [list(a) for a in zip(*self.data_processor.apply(datas, processes=processes, progress_bar=progress_bar))]

        self.preloaded_datas, self.preloaded_texts, self.preloaded_params = datas, texts, params

        if self.dataset.mode == DataSetMode.TRAIN and self.data_augmentation_amount > 0:
            abs_n_augs = int(self.data_augmentation_amount) if self.data_augmentation_amount >= 1 else int(self.data_augmentation_amount * len(self))
            self.preloaded_datas, self.preloaded_texts \
                = self.data_augmenter.augment_datas(datas, texts, n_augmentations=abs_n_augs,
                                                    processes=processes, progress_bar=progress_bar)

    def text_generator(self) -> Generator[str, None, None]:
        if len(self.preloaded_texts) > 0:
            for text in self.preloaded_texts:
                yield text
        else:
            for sample in self.dataset.samples():
                _, text = self.dataset.load_single_sample(sample, text_only=True)
                if self.text_processor:
                    text = self.text_processor.apply([text], 1, False)[0]
                yield text

    def generator(self) -> Generator[Tuple[np.array, List[str], Any], None, None]:
        if len(self.preloaded_datas) > 0:
            if self.dataset.mode == DataSetMode.TRAIN:
                # train mode wont generate parameters
                if self.generate_only_non_augmented:
                    # preloaded params store the 'length' of the non augmented data
                    for data, text, params in zip(self.preloaded_datas, self.preloaded_texts, self.preloaded_params):
                        yield data, text, None
                else:
                    for data, text in zip(self.preloaded_datas, self.preloaded_texts):
                        yield data, text, None
            else:
                # all other modes generate everything we got, but does not support data augmentation
                for data, text, params in zip(self.preloaded_datas, self.preloaded_texts, self.preloaded_params):
                    yield data, text, params

        else:
            data_aug_ratio = self.data_augmentation_amount if self.data_augmentation_amount < 1 else 1 - 1 / (self.data_augmentation_amount + 1)
            for sample in self.dataset.samples():
                line, text = self.dataset.load_single_sample(sample)
                if not self.dataset.is_sample_valid(sample, line, text):
                    if not self.skip_invalid_gt:
                        print("ERROR: invalid sample {}".format(sample))
                        continue

                if self.data_processor:
                    line, params = self.data_processor.apply([line], 1, False)[0]
                else:
                    params = None

                if self.text_processor:
                    text = self.text_processor.apply([text], 1, False)[0]

                if self.data_augmenter and np.random.rand() <= data_aug_ratio:
                    # data augmentation with given ratio
                    line, text = self.data_augmenter.augment_single(line, text)

                yield line, text, params





