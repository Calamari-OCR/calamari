import os
import zlib
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import tfaip.util.logging
from bidi.algorithm import get_base_level
from paiargparse import PAIArgumentParser, pai_meta, pai_dataclass

from calamari_ocr import __version__
from calamari_ocr.ocr.dataset import DataSetType
from calamari_ocr.ocr.dataset.datareader.base import CalamariDataGeneratorParams
from calamari_ocr.ocr.dataset.datareader.file import FileDataParams
from calamari_ocr.ocr.dataset.params import DATA_GENERATOR_CHOICES
from calamari_ocr.ocr.model.ctcdecoder.ctc_decoder import CTCDecoderParams, CTCDecoderType
from calamari_ocr.ocr.predict.params import Predictions, PredictorParams
from calamari_ocr.ocr.voting import VoterParams, VoterType
from calamari_ocr.utils.glob import glob_all

if TYPE_CHECKING:
    from calamari_ocr.ocr.dataset.pipeline import CalamariPipeline

logger = tfaip.util.logging.logger(__name__)


@pai_dataclass
@dataclass
class PredictArgs:
    checkpoint: List[str] = field(metadata=pai_meta(
        mode='flat',
        help="Path to the checkpoint without file extension"))
    data: CalamariDataGeneratorParams = field(default_factory=FileDataParams, metadata=pai_meta(
        mode='flat', choices=DATA_GENERATOR_CHOICES))
    extended_prediction_data: bool = field(default=False, metadata=pai_meta(
        mode='flat',
        help="Write: Predicted string, labels; position, probabilities and alternatives of chars to a .pred file"))
    extended_prediction_data_format: str = field(default='json', metadata=pai_meta(
        mode='flat',
        help="Extension format: Either pred or json. Note that json will not print logits."))
    no_progress_bars: bool = field(default=False, metadata=pai_meta(
        mode='flat',
        help="Do not show any progress bars"))
    ctc_decoder: CTCDecoderParams = field(default_factory=CTCDecoderParams)
    voter: VoterParams = field(default_factory=VoterParams)


def prepare_ctc_decoder_params(ctc_decoder: CTCDecoderParams):
    if ctc_decoder.dictionary:
        dictionary = set()
        logger.info("Creating dictionary")
        for path in glob_all(ctc_decoder.dictionary):
            with open(path, 'r') as f:
                dictionary = dictionary.union({word for word in f.read().split()})

        ctc_decoder.dictionary = dictionary
        logger.info("Dictionary with {} unique words successfully created.".format(len(dictionary)))

    if ctc_decoder.dictionary:
        logger.warning("USING A LANGUAGE MODEL IS CURRENTLY EXPERIMENTAL ONLY. NOTE: THE PREDICTION IS VERY SLOW!")
        ctc_decoder.type = CTCDecoderType.WordBeamSearch


def run(args: PredictArgs):
    # check if loading a json file
    # TODO: support running from JSON
    # if len(args.files) == 1 and args.files[0].endswith("json"):
    #     import json
    #     with open(args.files[0], 'r') as f:
    #        json_args = json.load(f)
    #        for key, value in json_args.items():
    #            setattr(args, key, value)

    # checks
    if args.extended_prediction_data_format not in ["pred", "json"]:
        raise Exception("Only 'pred' and 'json' are allowed extended prediction data formats")

    # add json as extension, resolve wildcard, expand user, ... and remove .json again
    args.checkpoint = [(cp if cp.endswith(".json") else cp + ".json") for cp in args.checkpoint]
    args.checkpoint = glob_all(args.checkpoint)
    args.checkpoint = [cp[:-5] for cp in args.checkpoint]

    # create ctc decoder
    prepare_ctc_decoder_params(args.ctc_decoder)

    # predict for all models
    from calamari_ocr.ocr.predict.predictor import MultiPredictor
    predictor = MultiPredictor.from_paths(checkpoints=args.checkpoint, voter_params=args.voter,
                                          predictor_params=PredictorParams(silent=True,
                                                                           progress_bar=not args.no_progress_bars))
    do_prediction = predictor.predict(predict_params)
    pipeline: CalamariPipeline = predictor.data.get_predict_data(predict_params)
    reader = pipeline.reader()
    if len(reader) == 0:
        raise Exception("Empty dataset provided. Check your files argument (got {})!".format(args.files))

    avg_sentence_confidence = 0
    n_predictions = 0

    reader.prepare_store()

    # output the voted results to the appropriate files
    for s in do_prediction:
        inputs, (result, prediction), meta = s.inputs, s.outputs, s.meta
        sample = reader.sample_by_id(meta['id'])
        n_predictions += 1
        sentence = prediction.sentence

        avg_sentence_confidence += prediction.avg_char_probability
        if args.verbose:
            lr = "\u202A\u202B"
            logger.info("{}: '{}{}{}'".format(meta['id'], lr[get_base_level(sentence)], sentence, "\u202C"))

        output_dir = args.output_dir

        reader.store_text_prediction(sentence, sample, output_dir=output_dir)

        if args.extended_prediction_data:
            ps = Predictions()
            ps.line_path = sample['image_path'] if 'image_path' in sample else sample['id']
            ps.predictions.extend([prediction] + [r.prediction for r in result])
            output_dir = output_dir if output_dir else os.path.dirname(ps.line_path)
            if not os.path.exists(output_dir):
                os.mkdir(output_dir)

            if args.extended_prediction_data_format == "pred":
                data = zlib.compress(ps.to_json(indent=2, ensure_ascii=False).encode('utf-8'))
            elif args.extended_prediction_data_format == "json":
                # remove logits
                for p in ps.predictions:
                    p.logits = None

                data = ps.to_json(indent=2)
            else:
                raise Exception("Unknown prediction format.")

            reader.store_extended_prediction(data, sample, output_dir=output_dir,
                                             extension=args.extended_prediction_data_format)

    logger.info("Average sentence confidence: {:.2%}".format(avg_sentence_confidence / n_predictions))

    reader.store(args.extension)
    logger.info("All prediction files written")


def main():
    parser = PAIArgumentParser()

    parser.add_argument('--version', action='version', version='%(prog)s v' + __version__)
    parser.add_root_argument("root", PredictArgs)

    parser.add_argument("--verbose", action="store_true",
                        help="Print additional information")
    parser.add_argument("--voter", type=str, default="confidence_voter_default_ctc",
                        help="The voting algorithm to use. Possible values: confidence_voter_default_ctc (default), "
                             "sequence_voter")
    parser.add_argument("--output_dir", type=str,
                        help="By default the prediction files will be written to the same directory as the given files. "
                             "You can use this argument to specify a specific output dir for the prediction files.")
    parser.add_argument("--dictionary", nargs="+", default=[],
                        help="List of text files that will be used to create a dictionary")
    parser.add_argument("--beam_width", type=int, default=25,
                        help='Number of beams when using the CTCWordBeamSearch decoder')

    args = parser.parse_args()

    run(args.root)


if __name__ == "__main__":
    main()
