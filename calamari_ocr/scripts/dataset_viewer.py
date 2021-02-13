import matplotlib.pyplot as plt

from paiargparse import PAIArgumentParser


from calamari_ocr import __version__
from calamari_ocr.ocr.dataset.data import Data
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGenerator
from calamari_ocr.ocr.dataset.imageprocessors import PrepareSample
from calamari_ocr.ocr.dataset.params import DataParams


def main():
    parser = PAIArgumentParser()
    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)

    parser.add_argument("--n_cols", type=int, default=1)
    parser.add_argument("--n_rows", type=int, default=5)
    parser.add_argument("--select", type=int, nargs="+", default=[])

    parser.add_argument("--preload", action='store_true', help='Simulate preloading')
    parser.add_argument("--as_validation", action='store_true', help="Access as validation instead of training data.")

    parser.add_root_argument("data", DataParams, Data.get_default_params())
    args = parser.parse_args()

    data_params: DataParams = args.data
    data_params.pre_proc.processors = list(p for p in data_params.pre_proc.processors if not isinstance(p, PrepareSample))  # NO PREPARE SAMPLE
    data_params.__post_init__()

    data = Data(data_params)
    data_pipeline = data.val_data() if args.as_validation else data.train_data()
    if not args.preload:
        reader: CalamariDataGenerator = data_pipeline.reader()
        if len(args.select) == 0:
            args.select = range(len(reader))
        else:
            reader._samples = [reader.samples()[i] for i in args.select]
    else:
        data.preload()
        data_pipeline = data.val_data() if args.as_validation else data.train_data()
        samples = data_pipeline.samples
        if len(args.select) == 0:
            args.select = range(len(samples))
        else:
            data_pipeline.samples = [samples[i] for i in args.select]

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


if __name__ == "__main__":
    main()
