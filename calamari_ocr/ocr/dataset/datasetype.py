from enum import IntEnum


# TODO: REMOVE THIS
class DataSetType(IntEnum):
    RAW = 0
    FILE = 1
    ABBYY = 2
    PAGEXML = 3
    HDF5 = 4
    EXTENDED_PREDICTION = 5
    GENERATED_LINE = 6

    @staticmethod
    def gt_extension(type):
        return {
            DataSetType.RAW: None,
            DataSetType.FILE: ".gt.txt",
            DataSetType.ABBYY: ".abbyy.xml",
            DataSetType.PAGEXML: ".xml",
            DataSetType.EXTENDED_PREDICTION: ".json",
            DataSetType.HDF5: ".h5",
            DataSetType.GENERATED_LINE: None,
        }[type]

    @staticmethod
    def pred_extension(type):
        return {
            DataSetType.RAW: None,
            DataSetType.FILE: ".pred.txt",
            DataSetType.ABBYY: ".pred.abbyy.xml",
            DataSetType.PAGEXML: ".pred.xml",
            DataSetType.EXTENDED_PREDICTION: ".json",
            DataSetType.HDF5: ".pred.h5",
            DataSetType.GENERATED_LINE: None,
        }[type]

