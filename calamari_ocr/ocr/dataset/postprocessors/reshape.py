from tfaip.base.data.pipeline.dataprocessor import DataProcessor


class ReshapeOutputsProcessor(DataProcessor):
    def apply(self, inputs, outputs, meta: dict):
        assert(inputs['img_len'].shape == (1,))
        assert(inputs['meta'].shape == (1,))
        inputs = inputs.copy()
        outputs = outputs.copy()
        inputs['img_len'] = inputs['img_len'][0]
        inputs['meta'] = inputs['meta'][0]
        for name in {'logits', 'softmax', 'blank_last_logits', 'blank_last_softmax'}:
            if name in outputs:
                outputs[name] = outputs[name][:outputs['out_len']]
        return inputs, outputs
