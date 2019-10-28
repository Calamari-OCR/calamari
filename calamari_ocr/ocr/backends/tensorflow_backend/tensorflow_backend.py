from calamari_ocr.ocr.backends.backend_interface import BackendInterface
from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_model import TensorflowModel


class TensorflowBackend(BackendInterface):
    def __init__(self,
                 checkpoint_params,
                 processes=-1):
        super().__init__(checkpoint_params)
        self.processes = processes if processes > 0 else 1

    def create_net(self, codec, graph_type, ctc_decoder_params=None, checkpoint_to_load=None, batch_size=-1, stream_input=True, codec_changes=None):
        model = TensorflowModel(self.network_proto, graph_type,
                                ctc_decoder_params=ctc_decoder_params,
                                batch_size=batch_size,
                                codec=codec,
                                processes=self.processes,
                                )

        if checkpoint_to_load:
            if codec_changes:
                # create a temporary model with the old weights, and copy the values including the codec changes
                source_model = TensorflowModel(checkpoint_to_load.checkpoint.model.network, graph_type, batch_size=batch_size)
                source_model.load_weights(checkpoint_to_load.ckpt_path)
                model.copy_weights_from_model(source_model, *codec_changes)
            else:
                model.load_weights(checkpoint_to_load.ckpt_path)

        return model


