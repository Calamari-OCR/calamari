from calamari_ocr.ocr.text_processing import text_processor_from_proto
from calamari_ocr.ocr.data_processing import data_processor_from_proto
from calamari_ocr.ocr import Codec
from calamari_ocr.ocr.backends import create_backend_from_proto
import time
import os
from tqdm import tqdm

from calamari_ocr.utils import RunningStatistics

from calamari_ocr.ocr import Predictor, Evaluator

from google.protobuf.json_format import MessageToJson

class Trainer:
    def __init__(self, checkpoint_params,
                 dataset,
                 validation_dataset=None,
                 txt_preproc=None,
                 txt_postproc=None,
                 data_preproc=None,
                 data_augmenter=None,
                 restore=None,
                 weights=None):
        self.checkpoint_params = checkpoint_params
        self.dataset = dataset
        self.validation_dataset = validation_dataset
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

        if self.validation_dataset:
            self.validation_dataset.load_samples(processes=checkpoint_params.processes, progress_bar=progress_bar)
            validation_datas, validation_txts = self.dataset.train_samples(skip_empty=checkpoint_params.skip_invalid_gt)
            if len(validation_datas) == 0:
                raise Exception("Validation dataset is empty. Provide valid validation data for early stopping.")
        else:
            validation_datas, validation_txts = [], []


        # preprocessing steps
        texts = self.txt_preproc.apply(txts, processes=checkpoint_params.processes, progress_bar=progress_bar)
        datas = self.data_preproc.apply(datas, processes=checkpoint_params.processes, progress_bar=progress_bar)
        validation_txts = self.txt_preproc.apply(validation_txts, processes=checkpoint_params.processes, progress_bar=progress_bar)
        validation_datas = self.data_preproc.apply(validation_datas, processes=checkpoint_params.processes, progress_bar=progress_bar)

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
        backend.set_train_data(datas, labels)
        backend.set_prediction_data(validation_datas)
        backend.prepare(train=True)

        loss_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.loss_stats)
        ler_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.ler_stats)
        dt_stats = RunningStatistics(checkpoint_params.stats_size, checkpoint_params.dt_stats)


        early_stopping_enabled = self.validation_dataset is not None \
                                 and checkpoint_params.early_stopping_frequency > 0 \
                                 and checkpoint_params.early_stopping_nbest > 1
        early_stopping_best_accuracy = checkpoint_params.early_stopping_best_accuracy
        early_stopping_best_cur_nbest = checkpoint_params.early_stopping_best_cur_nbest
        early_stopping_best_at_iter = checkpoint_params.early_stopping_best_at_iter

        # Start the actual training
        # ====================================================================================

        iter = checkpoint_params.iter

        # helper function to write a checkpoint
        def make_checkpoint(base_dir, prefix, version=None):
            if version:
                checkpoint_path = os.path.abspath(os.path.join(base_dir, "{}{}.ckpt".format(prefix, version)))
            else:
                checkpoint_path = os.path.abspath(os.path.join(base_dir, "{}{:08d}.ckpt".format(prefix, iter + 1)))
            print("Storing checkpoint to '{}'".format(checkpoint_path))
            backend.save_checkpoint(checkpoint_path)
            checkpoint_params.iter = iter
            checkpoint_params.loss_stats[:] = loss_stats.values
            checkpoint_params.ler_stats[:] = ler_stats.values
            checkpoint_params.dt_stats[:] = dt_stats.values
            checkpoint_params.total_time = time.time() - train_start_time
            checkpoint_params.early_stopping_best_accuracy = early_stopping_best_accuracy
            checkpoint_params.early_stopping_best_cur_nbest = early_stopping_best_cur_nbest
            checkpoint_params.early_stopping_best_at_iter = early_stopping_best_at_iter

            with open(checkpoint_path + ".json", 'w') as f:
                f.write(MessageToJson(checkpoint_params))

            return checkpoint_path

        try:
            # Training loop, can be interrupted by early stopping
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
                    make_checkpoint(checkpoint_params.output_dir, checkpoint_params.output_model_prefix)

                if early_stopping_enabled and (iter + 1) % checkpoint_params.early_stopping_frequency == 0:
                    print("Checking early stopping model")

                    if progress_bar:
                        out = list(tqdm(backend.prediction_step(checkpoint_params.batch_size),
                                        desc="Prediction",
                                        total=backend.num_prediction_steps(checkpoint_params.batch_size)))
                    else:
                        out = list(backend.prediction_step(checkpoint_params.batch_size))

                    pred_texts = [self.txt_postproc.apply("".join(codec.decode(d["decoded"]))) for d in out]
                    result = Evaluator.evaluate(gt_data=validation_txts, pred_data=pred_texts, progress_bar=progress_bar)
                    accuracy = 1 - result["avg_ler"]

                    if accuracy > early_stopping_best_accuracy:
                        early_stopping_best_accuracy = accuracy
                        early_stopping_best_cur_nbest = 1
                        early_stopping_best_at_iter = iter + 1
                        # overwrite as best model
                        make_checkpoint(
                            checkpoint_params.early_stopping_best_model_output_dir,
                            prefix="",
                            version=checkpoint_params.early_stopping_best_model_prefix,
                        )
                        print("Found better model with accuracy of {:%}".format(early_stopping_best_accuracy))
                    else:
                        early_stopping_best_cur_nbest += 1
                        print("No better model found. Currently accuracy of {:%} at iter {} (remaining nbest = {})".
                              format(early_stopping_best_accuracy, early_stopping_best_at_iter,
                                     checkpoint_params.early_stopping_nbest - early_stopping_best_cur_nbest))

                    if accuracy > 0 and early_stopping_best_cur_nbest >= checkpoint_params.early_stopping_nbest:
                        print("Early stopping now.")
                        break

        except KeyboardInterrupt as e:
            print("Storing interrupted checkpoint")
            make_checkpoint(checkpoint_params.output_dir,
                            checkpoint_params.output_model_prefix,
                            "interrupted")
            raise e

        print("Total time {}s for {} iterations.".format(time.time() - train_start_time, iter))


