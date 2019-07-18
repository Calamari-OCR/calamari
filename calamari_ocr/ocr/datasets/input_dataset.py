from .dataset import DataSet, DataSetMode, RawDataSet
from calamari_ocr.ocr.data_processing import DataPreprocessor
from calamari_ocr.ocr.text_processing import TextProcessor
from calamari_ocr.ocr.augmentation import DataAugmenter
from typing import Generator, Tuple, List, Any
import numpy as np
import multiprocessing
from collections import namedtuple
import queue
from calamari_ocr.utils.multiprocessing import tqdm_wrapper
from abc import ABC, abstractmethod
import logging

logger = logging.getLogger(__name__)


class OrderedQueueTask:
    def __init__(self, input_queue, output_queue, context=multiprocessing.get_context()):
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.context = context
        self.p = self.context.Process(daemon=True, target=self.run)

    def start(self):
        self.p.start()

    def stop(self):
        self.p.terminate()

    def join(self):
        self.p.join()

    def run(self) -> None:
        data = []
        current_idx = 0
        while True:
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

    def stop(self):
        self.p.terminate()

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


class InputDataset(ABC):
    def __init__(self,
                 mode: DataSetMode,
                 ):
        self.mode = mode
        self._generate_only_non_augmented = multiprocessing.Value('b', False)
        self.initialized = False

    def __enter__(self):
        if self.initialized:
            raise AssertionError("Input dataset already initialized.")

        logger.debug("InputDataset {} entered".format(self))
        self.initialized = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.initialized = False
        logger.debug("InputDataset {} exited".format(self))

    def check_initialized(self):
        if not self.initialized:
            raise AssertionError("InputDataset is not initialised. Call 'with InputDataset() as input_dataset:'. "
                                 "After the scope is closed the threads will be closed, too, for cleaning up.")

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def epoch_size(self):
        return len(self)

    @property
    def generate_only_non_augmented(self):
        return self._generate_only_non_augmented.value

    @generate_only_non_augmented.setter
    def generate_only_non_augmented(self, value):
        self._generate_only_non_augmented.value = value

    @abstractmethod
    def text_generator(self) -> Generator[str, None, None]:
        self.check_initialized()

    @abstractmethod
    def generator(self, epochs=1, text_only=False) -> Generator[Tuple[np.array, List[str], Any], None, None]:
        self.check_initialized()


class RawInputDataset(InputDataset):
    def __init__(self,
                 mode: DataSetMode,
                 raw_datas, raw_texts, raw_params,
                 ):
        super().__init__(mode)
        self.preloaded_datas, self.preloaded_texts, self.preloaded_params = raw_datas, raw_texts, raw_params

    def __len__(self):
        if self._generate_only_non_augmented.value:
            return len(self.preloaded_params)

        return len(self.preloaded_datas)

    def epoch_size(self):
        return len(self)

    def text_generator(self) -> Generator[str, None, None]:
        self.check_initialized()
        for text in self.preloaded_texts:
            yield text

    def generator(self, epochs=1, text_only=False) -> Generator[Tuple[np.array, List[str], Any], None, None]:
        self.check_initialized()
        for epoch in range(epochs):
            if self.mode == DataSetMode.TRAIN:
                # only train here, pred and eval are covered by else block
                # train mode wont generate parameters
                if self._generate_only_non_augmented.value:
                    # preloaded datas are ordered: first original data, then data augmented, however,
                    # preloaded params store the 'length' of the non augmented data
                    # thus, only orignal data is yielded
                    for data, text, params in zip(self.preloaded_datas, self.preloaded_texts, self.preloaded_params):
                        yield data, text, None
                else:
                    # yield all data, however no params
                    for data, text in zip(self.preloaded_datas, self.preloaded_texts):
                        yield data, text, None
            else:
                # all other modes generate everything we got, but does not support data augmentation
                for data, text, params in zip(self.preloaded_datas, self.preloaded_texts, self.preloaded_params):
                    yield data, text, params


class StreamingInputDataset(InputDataset):
    def __init__(self,
                 dataset: DataSet,
                 data_preprocessor: DataPreprocessor,
                 text_preprocessor: TextProcessor,
                 data_augmenter: DataAugmenter = None,
                 data_augmentation_amount: float = 0,
                 skip_invalid_gt=True,
                 processes=4):
        super().__init__(dataset.mode)
        self.dataset = dataset
        self.data_processor = data_preprocessor
        self.text_processor = text_preprocessor
        self.skip_invalid_gt = skip_invalid_gt
        self.data_augmenter = data_augmenter
        self.data_augmentation_amount = data_augmentation_amount
        self.mp_context = multiprocessing.get_context('spawn')
        self.processes = processes

        if data_augmenter and dataset.mode != DataSetMode.TRAIN and dataset.mode != DataSetMode.PRED_AND_EVAL:
            # no pred_and_eval bc it's augmentation
            raise Exception('Data augmentation is only supported for training, but got {} dataset instead'.format(dataset.mode))

        if data_augmentation_amount > 0 and self.data_augmenter is None:
            raise Exception('Requested data augmentation, but no data augmented provided. Use e. g. SimpleDataAugmenter')

        self.data_input_queue = None
        self.unordered_output_queue = None
        self.data_processing_tasks = []
        self.data_generator = None
        self.ordered_output_queue = None
        self.data_ordering = None

    def __enter__(self):
        super().__enter__()
        # create all tasks and queues
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
                    self._generate_only_non_augmented,
                ),
                self.data_input_queue,
                self.unordered_output_queue,
            ) for _ in range(self.processes)
        ]

        self.data_generator = self.dataset.create_generator(self.mp_context, self.data_input_queue)
        self.data_generator.start()
        self.ordered_output_queue = self.mp_context.JoinableQueue(self.processes * 4)
        self.data_ordering = OrderedQueueTask(self.unordered_output_queue, self.ordered_output_queue, self.mp_context)
        self.data_ordering.start()

        for p in self.data_processing_tasks:
            p.start()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # stop all tasks
        self.data_generator.stop()
        for p in self.data_processing_tasks:
            p.stop()
        self.data_ordering.stop()

        self.data_input_queue = None
        self.unordered_output_queue = None
        self.data_processing_tasks = []
        self.data_generator = None
        self.ordered_output_queue = None
        self.data_ordering = None

        super().__exit__(exc_type, exc_val, exc_tb)

    def __len__(self):
        return len(self.dataset.samples())

    def epoch_size(self):
        if self._generate_only_non_augmented.value:
            return len(self)

        if self.data_augmentation_amount >= 1:
            return int(len(self) * (1 + self.data_augmentation_amount))

        return int(1 / (1 - self.data_augmentation_amount) * len(self))

    def to_raw_input_dataset(self, processes=1, progress_bar=False, text_only=False) -> RawInputDataset:
        print("Preloading dataset type {} with size {}".format(self.dataset.mode, len(self)))
        prev = self._generate_only_non_augmented.value
        self._generate_only_non_augmented.value = True
        datas, texts, params = zip(*list(tqdm_wrapper(self.generator(epochs=1, text_only=text_only),
                                                      desc="Preloading data", total=len(self.dataset),
                                                      progress_bar=progress_bar)))
        preloaded_datas, preloaded_texts, preloaded_params = datas, texts, params
        self._generate_only_non_augmented.value = prev

        if (self.dataset.mode == DataSetMode.TRAIN or self.dataset.mode == DataSetMode.PRED_AND_EVAL) and self.data_augmentation_amount > 0:
            abs_n_augs = int(self.data_augmentation_amount) if self.data_augmentation_amount >= 1 else int(self.data_augmentation_amount * len(self))
            preloaded_datas, preloaded_texts \
                = self.data_augmenter.augment_datas(list(datas), list(texts), n_augmentations=abs_n_augs,
                                                    processes=processes, progress_bar=progress_bar)

        return RawInputDataset(self.mode, preloaded_datas, preloaded_texts, preloaded_params)

    def text_generator(self) -> Generator[str, None, None]:
        self.check_initialized()
        for _, text, _ in self.generator(epochs=1, text_only=True):
            if self.text_processor:
                text = self.text_processor.apply([text], 1, False)[0]
            yield text

    def generator(self, epochs=1, text_only=False) -> Generator[Tuple[np.array, List[str], Any], None, None]:
        self.check_initialized()
        self.data_generator.request(epochs, text_only)
        for epoch in range(epochs):
            for iter in range(len(self.dataset)):
                while True:
                    try:
                        global_id, id, line, text, params = self.ordered_output_queue.get(timeout=0.1)
                        yield line, text, params
                    except queue.Empty:
                        # test data ordering thread was canceled
                        if not self.data_ordering.p.is_alive() and self.ordered_output_queue.empty():
                            return
                        continue
                    except KeyboardInterrupt:
                        return

                    break
