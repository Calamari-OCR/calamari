from enum import Enum


class DataAugmentationAmountReference(Enum):
    ABSOLUTE = 'absolute'
    PERCENTAGE = 'relative'


# ABS: amount is the factor how often the dataset is augmented, e.g. 100 samples, amount == 2 => 200 augmented samples
#      in total 300 samples (=epoch size)
# PER: percentage is the probability to "chose" an augmented sample, e.g. 100 samples, percentage == 0.5 => 100
#      augmented samples, in total 200 samples (=epoch size). 100 / 200 = 0.5.
#      Or percentage == 0.75 => 300 augmented samplees, in total 400. 300 / 400 = 0.75
class DataAugmentationAmount:
    @staticmethod
    def from_factor(n):
        if n >= 1:
            return DataAugmentationAmount(DataAugmentationAmountReference.ABSOLUTE, 100, None)
        elif n > 0:
            return DataAugmentationAmount(DataAugmentationAmountReference.PERCENTAGE, None, n)
        elif n == 0:
            return DataAugmentationAmount(DataAugmentationAmountReference.PERCENTAGE, 0, 0)
        else:
            raise ValueError("Factor must be between (0, +infty) but got {}".format(n))

    def __init__(self,
                 reference=DataAugmentationAmountReference.ABSOLUTE,
                 amount=0,
                 percentage=0,
                 ):
        self.reference = reference
        self.amount = amount
        self.percentage = percentage

    def no_augs(self):
        return self.amount == 0 and self.percentage == 0

    def to_abs(self):
        if self.reference == DataAugmentationAmountReference.ABSOLUTE:
            return self.amount
        else:
            return int(1 / (1 - self.percentage) - 1)

    def to_rel(self):
        if self.reference == DataAugmentationAmountReference.ABSOLUTE:
            return 1 - 1 / (1 + self.amount)
        else:
            return self.percentage

    def epoch_size(self, n_samples: int):
        return int(n_samples * (self.to_abs() + 1))
