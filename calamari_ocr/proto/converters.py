import re

from calamari_ocr.proto.calamari_pb2 import LayerParams, NetworkParams


def default_network_params():
    params = NetworkParams()

    set_default_network_params(params)

    return params


def set_default_network_params(params):
    params.solver = NetworkParams.ADAM_SOLVER
    params.dropout = 0
    params.ctc_merge_repeated = True
    params.backend.cudnn = True
    params.learning_rate = 1e-3


def network_params_from_definition_string(str, params):
    cnn_matcher = re.compile(r"^([\d]+)(:([\d]+)(x([\d]+))?)?$")
    db_matcher = re.compile(r"^([\d]+):([\d]+)(:([\d]+)(x([\d]+))?)?$")
    concat_matcher = re.compile(r"^([\-\d]+):([\-\d]+)$")
    pool_matcher = re.compile(r"^([\d]+)(x([\d]+))?(:([\d]+)x([\d]+))?$")
    str_params = str.split(",")
    lstm_appeared = False
    set_default_network_params(params)
    for param in str_params:
        label, value = tuple(param.split("="))
        flags = ["ctc_merge_repeated", "cudnn"]
        floats = ["learning_rate", "momentum", "dropout"]
        if label in flags:
            setattr(params, label, value.lower() == "true")
        elif label == "ctc":
            setattr(params, label, NetworkParams.CTCType.Value(value))
        elif label == "l_rate":
            params.learning_rate = float(value)
        elif label in floats:
            setattr(params, label, float(value))
        elif label == "solver":
            params.solver = {
                "momentum": NetworkParams.MOMENTUM_SOLVER,
                "adam": NetworkParams.ADAM_SOLVER,
            }[value.lower()]
        elif label == "lstm":
            lstm_appeared = True
            layer = params.layers.add()
            layer.type = LayerParams.LSTM
            layer.lstm_direction = LayerParams.BIDIRECTIONAL_LSTM
            layer.hidden_nodes = int(value)
        elif label == "concat":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = concat_matcher.match(value)
            if match is None:
                raise Exception(
                    "Concat structure needs: concat=[index0]:[index1] but got concat={}".format(
                        value
                    )
                )

            match = match.groups()
            layer = params.layers.add()
            layer.type = LayerParams.CONCAT
            layer.concat_indices[:] = list(map(int, match))
        elif label == "db":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = db_matcher.match(value)
            if match is None:
                raise Exception(
                    "Dilated block structure needs: db=[filters]:[depth>0]:[h]x[w]"
                )

            match = match.groups()
            kernel_size = [2, 2]
            if match[2] is not None:
                kernel_size = [int(match[3])] * 2
            if match[4] is not None:
                kernel_size = [int(match[3]), int(match[5])]

            layer = params.layers.add()
            layer.type = LayerParams.DILATED_BLOCK
            layer.filters = int(match[0])
            layer.dilated_depth = int(match[1])
            layer.kernel_size.x = kernel_size[0]
            layer.kernel_size.y = kernel_size[1]
            layer.stride.x = 1
            layer.stride.y = 1
        elif label == "cnn":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = cnn_matcher.match(value)
            if match is None:
                raise Exception(
                    "CNN structure needs: cnn=[filters]:[h]x[w] but got {}".format(
                        value
                    )
                )

            match = match.groups()
            kernel_size = [2, 2]
            if match[1] is not None:
                kernel_size = [int(match[2])] * 2
            if match[3] is not None:
                kernel_size = [int(match[2]), int(match[4])]

            layer = params.layers.add()
            layer.type = LayerParams.CONVOLUTIONAL
            layer.filters = int(match[0])
            layer.kernel_size.x = kernel_size[0]
            layer.kernel_size.y = kernel_size[1]
            layer.stride.x = 1
            layer.stride.y = 1
        elif label == "tcnn":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")

            match = cnn_matcher.match(value)
            if match is None:
                raise Exception(
                    "Transposed CNN structure needs: tcnn=[filters]:[sx]x[sy]"
                )

            match = match.groups()
            kernel_size = [2, 2]
            stride = [2, 2]
            if match[1] is not None:
                stride = [int(match[2])] * 2
            if match[3] is not None:
                stride = [int(match[2]), int(match[4])]

            layer = params.layers.add()
            layer.type = LayerParams.TRANSPOSED_CONVOLUTIONAL
            layer.filters = int(match[0])
            layer.kernel_size.x = kernel_size[0]
            layer.kernel_size.y = kernel_size[1]
            layer.stride.x = stride[0]
            layer.stride.y = stride[1]
        elif label == "pool":
            if lstm_appeared:
                raise Exception("LSTM layers must be placed proceeding to CNN/Pool")
            match = pool_matcher.match(value)
            if match is None:
                raise Exception("Pool structure needs: pool=[h];[w]")

            match = match.groups()
            kernel_size = [int(match[0])] * 2
            if match[1] is not None:
                kernel_size = [int(match[0]), int(match[2])]

            if match[3] is not None:
                stride = [int(match[4]), int(match[5])]
            else:
                stride = kernel_size

            layer = params.layers.add()
            layer.type = LayerParams.MAX_POOLING
            layer.kernel_size.x = kernel_size[0]
            layer.kernel_size.y = kernel_size[1]
            layer.stride.x = stride[0]
            layer.stride.y = stride[1]
        else:
            raise Exception("Unknown layer with name: {}".format(label))

    return params
