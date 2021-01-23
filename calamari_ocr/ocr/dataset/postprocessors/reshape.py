from tfaip.base.data.pipeline.dataprocessor import DataProcessor
from tfaip.base.data.pipeline.definitions import Sample


class ReshapeOutputsProcessor(DataProcessor):
    def apply(self, sample: Sample) -> Sample:
        inputs = sample.inputs
        outputs = sample.outputs
        assert(inputs['img_len'].shape == (1,))
        assert(inputs['meta'].shape == (1,))
        inputs = inputs.copy()
        outputs = outputs.copy()
        inputs['img_len'] = inputs['img_len'][0]
        inputs['meta'] = inputs['meta'][0]

        def reshape_outputs(suffix):
            out_len = 'out_len' + suffix
            if out_len in outputs and outputs[out_len].shape == (1,):
                outputs[out_len] = outputs[out_len][0]

            for name in {'logits', 'softmax', 'blank_last_logits', 'blank_last_softmax'}:
                name += suffix
                if name in outputs:
                    outputs[name] = outputs[name][:outputs[out_len]]

        reshape_outputs('')
        for i in range(self.params.ensemble_):
            reshape_outputs(f"_{i}")

        return sample.new_inputs(inputs).new_outputs(outputs)
