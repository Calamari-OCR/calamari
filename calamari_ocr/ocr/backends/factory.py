from calamari_ocr.proto import BackendParams


def create_backend_from_proto(network_params, restore=None, weights=None, processes=-1):
    """
    Create a Backend implementation object based on NetworkParameters.

    Parameters
    ----------
    network_params : NetworkParameters
        the network parameters that define the new Backend
    restore : str
        path to a file to restore if a network is created
    weights : str
        path to a file to copy weights
    processes : int
        number of processes to use for all nets created by this backend.
        A negative number uses the default params as suggestest by the backend.

    Returns
    -------
        A net backend implementation object

    """
    # TODO: Change parameter to backend_params?
    # TODO: remove restore and weights
    if network_params.backend.type == BackendParams.TENSORFLOW:
        from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_backend import TensorflowBackend
        return TensorflowBackend(network_params, restore, weights, processes=processes)
    else:
        raise Exception("Unknown backend type '{}'".format(network_params.backend.type))

