from calamari_ocr.ocr.scenario import CalamariScenario


def rename(d, f, t):
    d[t] = d[f]
    del d[f]


def migrate_model_params(model: dict):
    convert_layer_name = {
        'convolutional': "calamari_ocr.ocr.model.layers.conv2d:Conv2DLayerParams",
        'concat': "calamari_ocr.ocr.model.layers.concat:ConcatLayerParams",
        'max_pooling': "calamari_ocr.ocr.model.layers.pool2d:MaxPool2DLayerParams",
        'lstm': "calamari_ocr.ocr.model.layers.bilstm:BiLSTMLayerParams",
        'transposed_conv': "calamari_ocr.ocr.model.layers.transposedconv2d:TransposedConv2DLayerParams",
        'dilated_block': "calamari_ocr.ocr.model.layers.dilatedblock:DilatedBlockLayerParams",
        'dropout': "calamari_ocr.ocr.model.layers.dropout:DropoutLayerParams",
    }
    for layer in model['layers']:
        del layer['lstm_direction']
        if layer['type'] != 'lstm':
            del layer['peepholes']
            del layer['hidden_nodes']
        if layer['type'] != 'dilated_block':
            del layer['dilated_depth']
        if layer['type'] != 'concat':
            del layer['concat_indices']
        if layer['type'] not in {'convolutional', 'max_pooling', 'dilated_block', 'transposed_conv'}:
            del layer['stride']
            del layer['filters']
            del layer['kernel_size']
        if layer['type'] == 'max_pooling':
            del layer['filters']
            rename(layer, 'kernel_size', 'pool_size')

        layer['__cls__'] = convert_layer_name[layer['type']]
        del layer['type']

    model['layers'].append({
        '__cls__': convert_layer_name['dropout'],
        'rate': model['dropout']
    })
    del model['dropout']


def migrate(trainer_params: dict) -> dict:
    convert_processor_name = {
        "CenterNormalizer": "calamari_ocr.ocr.dataset.imageprocessors.center_normalizer:CenterNormalizerProcessorParams",
        "DataRangeNormalizer": "calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer:DataRangeProcessorParams",
        "FinalPreparation": "calamari_ocr.ocr.dataset.imageprocessors.final_preparation:FinalPreparationProcessorParams",
        "AugmentationProcessor": "calamari_ocr.ocr.dataset.imageprocessors.augmentation:AugmentationProcessorParams",
        "StripTextProcessor": "calamari_ocr.ocr.dataset.textprocessors.basic_text_processors:StripTextProcessorParams",
        "TextNormalizer": "calamari_ocr.ocr.dataset.textprocessors.text_normalizer:TextNormalizerProcessorParams",
        "TextRegularizer": "calamari_ocr.ocr.dataset.textprocessors.text_regularizer:TextRegularizerProcessorParams",
        "PrepareSampleProcessor": "calamari_ocr.ocr.dataset.imageprocessors.preparesample:PrepareSampleProcessorParams",
        "ReshapeOutputsProcessor": "calamari_ocr.ocr.dataset.postprocessors.reshape:ReshapeOutputsProcessorParams",
        "CTCDecoderProcessor": "calamari_ocr.ocr.dataset.postprocessors.ctcdecoder:CTCDecoderProcessorParams",
    }

    for name in ['scenario']:
        rename(trainer_params, name + '_params', name)

    for name in ['model', 'data']:
        rename(trainer_params['scenario'], name + '_params', name)

    scenario = trainer_params['scenario']
    scenario['data']['__cls__'] = 'calamari_ocr.ocr.dataset.params:DataParams'
    scenario['model']["__cls__"] = "calamari_ocr.ocr.model.params:ModelParams"

    data = scenario['data']
    rename(data, 'line_height_', 'line_height')
    rename(data, 'skip_invalid_gt_', 'skip_invalid_gt')
    rename(data, 'resource_base_path_', 'resource_base_path')
    rename(data, 'pre_processors_', 'pre_proc')
    rename(data, 'post_processors_', 'post_proc')
    rename(data['pre_proc'], 'sample_processors', 'processors')
    rename(data['post_proc'], 'sample_processors', 'processors')
    data['post_proc']['__cls__'] = "tfaip.base.data.pipeline.processor.params:SequentialProcessorPipelineParams"
    data['pre_proc']['__cls__'] = "tfaip.base.data.pipeline.processor.params:SequentialProcessorPipelineParams"
    for proc in data['pre_proc']['processors'] + data['post_proc']['processors']:
        if 'args' in proc:
            args = proc['args']
            if args:
                for k, v in args.items():
                    proc[k] = v
            del proc['args']
        name = proc['name']
        del proc['name']
        proc['__cls__'] = convert_processor_name[name]

    migrate_model_params(scenario['model'])

    CalamariScenario.params_from_dict(scenario)
    return {'scenario': scenario}
