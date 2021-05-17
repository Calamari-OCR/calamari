from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Type, TypeVar, Generic

import numpy as np
from paiargparse import pai_dataclass
from tfaip.util.multiprocessing.parallelmap import parallel_map

from calamari_ocr.utils.image import load_image


@pai_dataclass
@dataclass
class DataAugmenterParams(ABC):
    @classmethod
    @abstractmethod
    def cls(cls) -> Type["DataAugmenterBase"]:
        raise NotImplementedError

    def create(self) -> "DataAugmenterBase":
        return self.cls()(self)


TDataAugParams = TypeVar("TDataAugParams", bound=DataAugmenterParams)


class DataAugmenterBase(Generic[TDataAugParams], ABC):
    def __init__(self, params: TDataAugParams):
        super().__init__()
        self.params = params

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
        if n_augmentations < 0 or not isinstance(n_augmentations, int):
            raise ValueError("Number of augmentation must be an integer >= 0")

        if n_augmentations == 0:
            return datas, gt_txts

        out = parallel_map(
            self.augment_data_tuple,
            list(zip(datas, gt_txts, [n_augmentations] * len(datas))),
            desc="Augmentation",
            processes=processes,
            progress_bar=progress_bar,
        )
        out_d, out_t = type(datas)(), type(datas)()
        for d, t in out:
            out_d += d
            out_t += t

        return datas + out_d, gt_txts + out_t


@pai_dataclass(alt="Simple")
@dataclass
class DefaultDataAugmenterParams(DataAugmenterParams):
    @classmethod
    def cls(cls) -> Type["DataAugmenterBase"]:
        return DefaultDataAugmenter


class DefaultDataAugmenter(DataAugmenterBase[DefaultDataAugmenterParams]):
    def augment_single(self, data, gt_txt):
        import calamari_ocr.thirdparty.ocrodeg as ocrodeg

        original_dtype = data.dtype
        data = data.astype(np.float)
        m = data.max()
        data = data / (1 if m == 0 else m)
        data = ocrodeg.random_pad(data, (0, data.shape[1] * 2))
        # data = ocrodeg.transform_image(data, **ocrodeg.random_transform(rotation=(-0.1, 0.1), translation=(-0.1, 0.1)))
        for sigma in [2, 5]:
            noise = ocrodeg.bounded_gaussian_noise(data.shape, sigma, 3.0)
            data = ocrodeg.distort_with_noise(data, noise)

        data = ocrodeg.printlike_multiscale(data, blur=1, inverted=True)
        data = (data * 255 / data.max()).astype(original_dtype)
        return data, gt_txt


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    aug = DefaultDataAugmenterParams().create()
    img = 255 - np.mean(
        load_image("../../test/data/uw3_50lines/train/010001.bin.png")[:, :, 0:2],
        axis=-1,
    )
    aug_img = [aug.augment_single(img.T, "")[0].T for _ in range(4)]
    f, ax = plt.subplots(5, 1)
    ax[0].imshow(255 - img, cmap="gray")
    for i, x in enumerate(aug_img):
        ax[i + 1].imshow(255 - x, cmap="gray")
    plt.show()
