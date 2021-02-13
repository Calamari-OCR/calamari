from dataclasses import field, dataclass

import tfaip.util.logging as logging
from paiargparse import PAIArgumentParser, pai_meta, pai_dataclass
from tfaip import PipelineMode
from tfaip.data.databaseparams import DataPipelineParams

from calamari_ocr import __version__
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.imageprocessors import PrepareSampleProcessorParams, AugmentationProcessorParams
from calamari_ocr.ocr.dataset.params import DataParams, DATA_GENERATOR_CHOICES

logger = logging.logger(__name__)


@pai_dataclass
@dataclass
class DataWrapper:
    data: DataParams = field(default_factory=Data.default_params, metadata=pai_meta(
        fix_dc=True, mode='flat'
    ))
    gen: CalamariDataGeneratorParams = field(default_factory=FileDataParams, metadata=pai_meta(
        mode='flat', choices=DATA_GENERATOR_CHOICES,
    ))
    pipeline: DataPipelineParams = field(default_factory=DataPipelineParams, metadata=pai_meta(
        mode='flat', fix_dc=True
    ))


def main(args=None):
    parser = PAIArgumentParser()
    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

    parser.add_argument("--n_cols", type=int, default=1)
    parser.add_argument("--n_rows", type=int, default=5)
    parser.add_argument("--select", type=int, nargs="+", default=[])

    parser.add_argument("--preload", action='store_true', help='Simulate preloading')
    parser.add_argument("--as_validation", action='store_true', help="Access as validation instead of training data.")
    parser.add_argument("--n_augmentations", type=float, default=0)
    parser.add_argument("--no_plot", action='store_true', help='This parameter is for testing only')

    parser.add_root_argument("data", DataWrapper)
    args = parser.parse_args(args=args)

    data_wrapper: DataWrapper = args.data
    data_params = data_wrapper.data
    data_params.pre_proc.run_parallel = False
    data_params.pre_proc.erase_all(PrepareSampleProcessorParams)
    for p in data_params.pre_proc.processors_of_type(AugmentationProcessorParams):
        p.n_augmentations = args.n_augmentations
    data_params.__post_init__()
    data_wrapper.pipeline.mode = PipelineMode.EVALUATION if args.as_validation else PipelineMode.TRAINING
    data_wrapper.gen.prepare_for_mode(data_wrapper.pipeline.mode)

    data = Data(data_params)
    if len(args.select) == 0:
        args.select = list(range(len(data_wrapper.gen)))
    else:
        try:
            data_wrapper.gen.select(args.select)
        except NotImplementedError:
            logger.warning(f"Selecting is not supported for a data generator of type {type(data_wrapper.gen)}. "
                           f"Resuming without selection.")
    data_pipeline = data.create_pipeline(data_wrapper.pipeline, data_wrapper.gen)
    if args.preload:
        data_pipeline = data_pipeline.as_preloaded()

    if args.no_plot:
        with data_pipeline as dataset:
            list(zip(args.select, dataset.generate_input_samples(auto_repeat=False)))
        return

    import matplotlib.pyplot as plt
    f, ax = plt.subplots(args.n_rows, args.n_cols, sharey='all')
    row, col = 0, 0
    with data_pipeline as dataset:
        for i, (id, sample) in enumerate(zip(args.select, dataset.generate_input_samples(auto_repeat=False))):
            line, text, params = sample.inputs, sample.targets, sample.meta
            if args.n_cols == 1:
                ax[row].imshow(line.transpose())
                ax[row].set_title("ID: {}\n{}".format(id, text))
            else:
                ax[row, col].imshow(line.transpose())
                ax[row, col].set_title("ID: {}\n{}".format(id, text))

            row += 1
            if row == args.n_rows:
                row = 0
                col += 1

            if col == args.n_cols or i == len(dataset) - 1:
                plt.show()
                f, ax = plt.subplots(args.n_rows, args.n_cols, sharey='all')
                row, col = 0, 0

    plt.show()


if __name__ == "__main__":
    main()
