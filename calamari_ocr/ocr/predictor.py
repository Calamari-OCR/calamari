import time

from tqdm import tqdm

from google.protobuf import json_format

from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
from calamari_ocr.proto import CheckpointParams


class Predictor:
    def __init__(self, checkpoint=None, text_postproc=None, data_preproc=None):
        if checkpoint:
            with open(checkpoint + '.json', 'r') as f:
                checkpoint_params = json_format.Parse(f.read(), CheckpointParams())
                self.model_params = checkpoint_params.model
        else:
            raise Exception("No checkpoint provided.")

        self.checkpoint = checkpoint
        self.text_postproc = text_postproc if text_postproc else text_processor_from_proto(self.model_params.text_postprocessor, "post")
        self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(self.model_params.data_preprocessor)

    def predict(self, dataset, batch_size=1, processes=1, progress_bar=True):
        start_time = time.time()

        dataset.load_samples(processes=processes, progress_bar=progress_bar)
        datas = dataset.prediction_samples()

        # preprocessing steps
        datas = self.data_preproc.apply(datas, processes=processes, progress_bar=progress_bar)

        codec = Codec(self.model_params.codec.charset)

        # create backend
        network_params = self.model_params.network

        backend = create_backend_from_proto(network_params, restore=self.checkpoint)
        backend.set_data(datas)
        backend.prepare(train=False)

        prediction_start_time = time.time()

        if progress_bar:
            out = list(tqdm(backend.prediction_step(batch_size), desc="Prediction", total=backend.num_prediction_steps(batch_size)))
        else:
            out = list(backend.prediction_step(batch_size))

        for d in out:
            d['sentence'] = self.text_postproc.apply("".join(codec.decode(d["decoded"])))

        print("Total time: {}s, Prediction time: {}s, i. e. {}s per line.".format(
            time.time() - start_time, time.time() - prediction_start_time,
            (time.time() - prediction_start_time) / len(out)))



        return out, dataset.samples(), codec
