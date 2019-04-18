from .dataset import DataSet, DataSetMode, RawDataSet
from calamari_ocr.ocr.data_processing import DataPreprocessor
from calamari_ocr.ocr.text_processing import TextProcessor
from calamari_ocr.ocr.augmentation import DataAugmenter
from typing import Generator, Tuple, List, Any, NamedTuple
import numpy as np
import multiprocessing
from collections import namedtuple
import queue


class OrderedQueueTask:
    def __init__(self, input_queue, output_queue, total, context=multiprocessing.get_context()):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.total = total
        self.context = context
        self.p = self.context.Process(daemon=True, target=self.run)

    def start(self):
        self.p.start()

    def join(self):
        self.p.join()

    def run(self) -> None:
        data = []
        current_idx = 0
        for i in range(self.total):
            while True:
                try:
                    data.append(self.input_queue.get(timeout=0.1))
                except queue.Empty:
                    continue
                except KeyboardInterrupt:
                    return

                break

            data.sort(key=lambda data: data[0])
            while len(data) > 0 and data[0][0] <= current_idx:
                try:
                    self.output_queue.put(data[0], timeout=0.1)
                    self.output_queue.task_done()
                    del data[0]
                    current_idx += 1
                except queue.Full:
                    continue
                except KeyboardInterrupt:
                    return


DataProcessingTaskData = namedtuple("DataProcessingTaskData", [
    "skip_invalid_gt",
    "data_aug_ratio",
    "text_processor",
    "data_processor",
    "data_augmenter",
    "generate_only_non_augmented",
])


class DataProcessingTask:
    def __init__(self, params, input_queue: multiprocessing.JoinableQueue, output_queue: multiprocessing.JoinableQueue, context=multiprocessing.get_context()):
        self.params = params
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.p = context.Process(daemon=True, target=self.run)

    def start(self):
        self.p.start()

    def join(self):
        self.p.join()

    def run(self) -> None:
        while True:
            try:
                data = self.input_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            except KeyboardInterrupt:
                # allow keyboard interrupt
                return

            out = self.apply_single(*data)
            if out:
                while True:
                    try:
                        self.output_queue.put(out, timeout=0.1)
                        break
                    except queue.Full:
                        continue
                    except KeyboardInterrupt:
                        return

            self.output_queue.task_done()

    def apply_single(self, idx, sample_id, line, text):
        #if not dataset.is_sample_valid(sample, line, text):
        #    if not skip_invalid_gt:
        #        print("ERROR: invalid sample {}".format(sample))
        #        return None

        if self.params.data_processor and line is not None:
            line, params = self.params.data_processor.apply([line], 1, False)[0]
        else:
            params = None

        if self.params.text_processor and text is not None:
            text = self.params.text_processor.apply([text], 1, False)[0]

        if line is not None and not self.params.generate_only_non_augmented.value and self.params.data_augmenter and np.random.rand() <= self.params.data_aug_ratio:
            # data augmentation with given ratio
            line, text = self.params.data_augmenter.augment_single(line, text)

        return idx, sample_id, line, text, params


def RawInputDataset(
        mode: DataSetMode,
        raw_datas, raw_texts, raw_params,
        data_preprocessor, text_preprocessor,
        data_augmenter=None, data_augmentation_amount=0
):
    dataset = InputDataset(RawDataSet(mode=mode, images=raw_datas, texts=raw_texts),
                           data_preprocessor,
                           text_preprocessor,
                           data_augmenter, data_augmentation_amount,
                           processes=0,
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
                 skip_invalid_gt=True,
                 processes=4):
        self.dataset = dataset
        self.data_processor = data_preprocessor
        self.text_processor = text_preprocessor
        self.skip_invalid_gt = skip_invalid_gt
        self.data_augmenter = data_augmenter
        self.preloaded_datas = []
        self.preloaded_texts = []
        self.preloaded_params = []
        self.data_augmentation_amount = data_augmentation_amount
        self.generate_only_non_augmented = multiprocessing.Value('b', False)
        self.mp_context = multiprocessing.get_context('spawn')
        self.processes = processes

        if data_augmenter and dataset.mode != DataSetMode.TRAIN:
            raise Exception('Data augmentation is only supported for training, but got {} dataset instead'.format(dataset.mode))

        if data_augmentation_amount > 0 and self.data_augmenter is None:
            raise Exception('Requested data augmentation, but no data augmented provided. Use e. g. SimpleDataAugmenter')

        self.data_input_queue = self.mp_context.JoinableQueue(self.processes * 4)
        self.unordered_output_queue = self.mp_context.JoinableQueue()

        self.data_processing_tasks = [
            DataProcessingTask(
                DataProcessingTaskData(
                    self.skip_invalid_gt,
                    self.data_augmentation_amount if self.data_augmentation_amount < 1 else 1 - 1 / (self.data_augmentation_amount + 1),
                    self.text_processor,
                    self.data_processor,
                    self.data_augmenter,
                    self.generate_only_non_augmented,
                ),
                self.data_input_queue,
                self.unordered_output_queue,
            ) for _ in range(processes)
        ]

        for p in self.data_processing_tasks:
            p.start()

    def __len__(self):
        return len(self.dataset.samples())

    def epoch_size(self):
        if self.generate_only_non_augmented:
            return len(self)

        if self.data_augmentation_amount >= 1:
            return int(len(self) * self.data_augmentation_amount)

        return int(1 / (1 - self.data_augmentation_amount) * len(self))

    def preload(self, processes=1, progress_bar=False, text_only=False):
        print("Preloading dataset type {} with size {}".format(self.dataset.mode, len(self)))
        self.generate_only_non_augmented.value = True
        datas, texts, params = zip(*list(self.generator(epochs=1, text_only=text_only)))
        self.preloaded_datas, self.preloaded_texts, self.preloaded_params = datas, texts, params

        if self.dataset.mode == DataSetMode.TRAIN and self.data_augmentation_amount > 0:
            abs_n_augs = int(self.data_augmentation_amount) if self.data_augmentation_amount >= 1 else int(self.data_augmentation_amount * len(self))
            self.preloaded_datas, self.preloaded_texts \
                = self.data_augmenter.augment_datas(list(datas), list(texts), n_augmentations=abs_n_augs,
                                                    processes=processes, progress_bar=progress_bar)

    def text_generator(self) -> Generator[str, None, None]:
        if len(self.preloaded_texts) > 0:
            for text in self.preloaded_texts:
                yield text
        else:
            for _, text, _ in self.generator(epochs=1, text_only=True):
                if self.text_processor:
                    text = self.text_processor.apply([text], 1, False)[0]
                yield text

    def generator(self, epochs=1, text_only=False) -> Generator[Tuple[np.array, List[str], Any], None, None]:
        if len(self.preloaded_datas) > 0:
            for epoch in range(epochs):
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
            # create a generator of the dataset that enqueues loaded samples
            # the samples are processed by the data preprocessing and data augmenter queues and written unordered
            # the data ordering queue and process yield a sorted output of lines
            total = epochs * len(self.dataset)
            data_generator = self.dataset.create_generator(self.data_input_queue, epochs, text_only=text_only)
            data_generator.start()
            ordered_output_queue = self.mp_context.JoinableQueue(self.processes * 4)
            data_ordering = OrderedQueueTask(self.unordered_output_queue, ordered_output_queue, total, self.mp_context)
            data_ordering.start()

            for epoch in range(epochs):
                for iter in range(len(self.dataset)):
                    while True:
                        try:
                            global_id, id, line, text, params = ordered_output_queue.get(timeout=0.1)
                            yield line, text, params
                        except queue.Empty:
                            # test data ordering thread was canceled
                            if not data_ordering.p.is_alive() and ordered_output_queue.empty():
                                return
                            continue
                        except KeyboardInterrupt:
                            return

                        break

            data_generator.join()
            data_ordering.join()
