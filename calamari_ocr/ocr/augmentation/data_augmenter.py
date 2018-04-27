from abc import ABC, abstractmethod
from calamari_ocr.utils import parallel_map


class DataAugmenter(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def augment_single(self, data, gt_txt):
        pass

    def augment_data(self, data, gt_txt, n_augmentations):
        if n_augmentations <= 0:
            return data, gt_txt

        return zip(*[self.augment_single(data, gt_txt) for _ in range(n_augmentations)])

    def augment_data_tuple(self, t):
        return self.augment_data(*t)

    def augment_datas(self, datas, gt_txts, n_augmentations, processes=1, progress_bar=False):
        if n_augmentations <= 0:
            return datas, gt_txts

        out = parallel_map(self.augment_data_tuple, list(zip(datas, gt_txts, [n_augmentations] * len(datas))),
                           desc="Augmentation", processes=processes, progress_bar=progress_bar)
        out_d, out_t = [], []
        for d, t in out:
            out_d += d
            out_t += t

        return datas + out_d, gt_txts + out_t


class NoopDataAugmenter(DataAugmenter):
    def __init__(self):
        super().__init__()

    def augment_single(self, data, gt_txt):
        return data, gt_txt

    def augment_data(self, data, gt_txt, n_augmentations):
        return data * n_augmentations, gt_txt * n_augmentations


class SimpleDataAugmenter(DataAugmenter):
    def __init__(self):
        super().__init__()

    def augment_single(self, data, gt_txt):
        import calamari_ocr.thirdparty.ocrodeg as ocrodeg
        data = ocrodeg.random_pad(data, (0, data.shape[1] * 5))
        for sigma in [2, 5]:
            noise = ocrodeg.bounded_gaussian_noise(data.shape, sigma, 3.0)
            data = ocrodeg.distort_with_noise(data, noise)

        return ocrodeg.printlike_multiscale(data, inverted=True), gt_txt
