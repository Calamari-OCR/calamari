from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
import time
import os

from calamari_ocr.utils import RunningStatistics

from google.protobuf.json_format import MessageToJson

class Trainer:
    def __init__(self, checkpoint_params,
                 dataset,
                 txt_preproc=None,
                 txt_postproc=None,
                 data_preproc=None,
                 data_augmenter=None,
                 restore=None,
                 weights=None):
        self.checkpoint_params = checkpoint_params
        self.dataset = dataset
        self.data_augmenter = data_augmenter
        self.txt_preproc = txt_preproc if txt_preproc else text_processor_from_proto(checkpoint_params.model.text_preprocessor, "pre")
        self.txt_postproc = txt_postproc if txt_postproc else text_processor_from_proto(checkpoint_params.model.text_postprocessor, "post")
        self.data_preproc = data_preproc if data_preproc else data_processor_from_proto(checkpoint_params.model.data_preprocessor)
        self.restore = restore
        self.weights = weights

    def train(self, progress_bar=False):
        checkpoint_params = self.checkpoint_params

        train_start_time = time.time() + self.checkpoint_params.total_time

        self.dataset.load_samples(processes=checkpoint_params.processes, progress_bar=progress_bar)
        datas, txts = self.dataset.train_samples(skip_empty=checkpoint_params.skip_invalid_gt)

        if len(datas) == 0:
            raise Exception("Empty dataset is not allowed. Check if the data is at the correct location")

        # preprocessing steps
        texts = self.txt_preproc.apply(txts, processes=checkpoint_params.processes, progress_bar=progress_bar)
        datas = self.data_preproc.apply(datas, processes=checkpoint_params.processes, progress_bar=progress_bar)

        # compute the codec
        codec = Codec.from_texts(texts)
        checkpoint_params.model.codec.charset[:] = codec.charset


        # TODO: Data augmentation

        # compute the labels
        labels = [codec.encode(txt) for txt in texts]

        # create backend
        network_params = checkpoint_params.model.network
        network_params.features = checkpoint_params.model.line_height
        network_params.classes = len(codec)

        backend = create_backend_from_proto(network_params, restore=self.restore, weights=self.weights)
        backend.set_data(datas, labels)
        backend.prepare(train=True)

        loss_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.loss_stats)
        ler_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.ler_stats)
        dt_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.dt_stats)

        try:
            iter = checkpoint_params.iter

            def make_checkpoint(version=None):
                if version:
                    checkpoint_path = os.path.abspath("{}{}.ckpt".format(checkpoint_params.output_path_prefix, version))
                else:
                    checkpoint_path = os.path.abspath("{}{:08d}.ckpt".format(checkpoint_params.output_path_prefix, iter + 1))
                print("Storing checkpoint to '{}'".format(checkpoint_path))
                backend.save_checkpoint(checkpoint_path)
                checkpoint_params.iter = iter
                checkpoint_params.loss_stats[:] = loss_stats.values
                checkpoint_params.ler_stats[:] = ler_stats.values
                checkpoint_params.dt_stats[:] = dt_stats.values
                checkpoint_params.total_time = time.time() - train_start_time

                with open(checkpoint_path + ".json", 'w') as f:
                    f.write(MessageToJson(checkpoint_params))

            for iter in range(iter, checkpoint_params.max_iters):
                checkpoint_params.iter = iter

                iter_start_time = time.time()
                result = backend.train_step(checkpoint_params.batch_size)
                loss_stats.push(result['loss'])
                ler_stats.push(result['ler'])

                dt_stats.push(time.time() - iter_start_time)

                if iter % checkpoint_params.display == 0:
                    pred_sentence = self.txt_postproc.apply("".join(codec.decode(result["decoded"][0])))
                    gt_sentence = self.txt_postproc.apply("".join(codec.decode(result["gt"][0])))
                    print("#{:08d}: loss={:.8f} ler={:.8f} dt={:.8f}s".format(iter, loss_stats.mean(), ler_stats.mean(), dt_stats.mean()))
                    print(" PRED: '{}'".format(pred_sentence))
                    print(" TRUE: '{}'".format(gt_sentence))

                if (iter + 1) % checkpoint_params.checkpoint_frequency == 0:
                    make_checkpoint()

        except KeyboardInterrupt as e:
            print("Storing interrupted checkpoint")
            make_checkpoint("interrupted")
            raise e

        print("Total time {}s for {} iterations.".format(time.time() - train_start_time, iter))


