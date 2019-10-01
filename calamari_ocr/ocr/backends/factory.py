def create_backend_from_checkpoint(checkpoint_params, restore=None, weights=None, processes=-1):
    """
    Create a Backend implementation object based on NetworkParameters.

    Parameters
    ----------
    checkpoint_params : any
        the checkpoint to load
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
    from calamari_ocr.ocr.backends.tensorflow_backend.tensorflow_backend import TensorflowBackend
    return TensorflowBackend(checkpoint_params, processes=processes)

