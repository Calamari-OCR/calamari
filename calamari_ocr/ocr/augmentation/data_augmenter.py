from abc import ABC, abstractmethod

class DataAugmenter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def augment_single(self, data, gt_txt):
        pass

    @abstractmethod
    def augment_data(self, data, gt_txt, n_augmentations):
        pass


class NoopDataAugmenter(DataAugmenter):
    def __init__(self):
        super().__init__()

    def augment_single(self, data, gt_txt):
        return data, gt_txt

    def augment_data(self, data, gt_txt, n_augmentations):
        return data * n_augmentations, gt_txt * n_augmentations
