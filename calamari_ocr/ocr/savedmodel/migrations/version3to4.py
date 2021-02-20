from calamari_ocr.ocr.scenario import CalamariScenario



def migrate(trainer_params: dict) -> dict:
    convert_processor_name = {
        "CenterNormalizer": "calamari_ocr.ocr.dataset.imageprocessors.center_normalizer:CenterNormalizerParams",
        "DataRangeNormalizer": "calamari_ocr.ocr.dataset.imageprocessors.data_range_normalizer:DataRangeProcessorParams",
        "FinalPreparation": "calamari_ocr.ocr.dataset.imageprocessors.final_preparation:FinalPreparationProcessorParams",
        "AugmentationProcessor": "calamari_ocr.ocr.dataset.imageprocessors.augmentation:AugmentationParams",
        "StripTextProcessor": "calamari_ocr.ocr.dataset.textprocessors.basic_text_processors:StripText",
        "TextNormalizer": "calamari_ocr.ocr.dataset.textprocessors.text_normalizer:TextNormalizer",
        "TextRegularizer": "calamari_ocr.ocr.dataset.textprocessors.text_regularizer:TextRegularizer",
        "PrepareSampleProcessor": "calamari_ocr.ocr.dataset.imageprocessors.preparesample:PrepareSampleProcessorParams",
        "ReshapeOutputsProcessor": "calamari_ocr.ocr.dataset.postprocessors.reshape:ReshapeOutputs",
        "CTCDecoderProcessor": "calamari_ocr.ocr.dataset.postprocessors.ctcdecoder:CTCDecoder",
    }

    def rename(d, f, t):
        d[t] = d[f]
        del d[f]

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

    CalamariScenario.params_from_dict(scenario)
    return scenario
