Predicting
==========

First you need to create a predictor object giving an existing model.

Python API
----------
The predictor must be created once but can then be used for multiple predictions.

Single Model
~~~~~~~~~~~~

There are two options for prediction depending on your data and usecase.

If you have already a `list of data` use the code below: create the predictor and call ``predict_raw``, ``predict_pipeline``, ``predict_dataset``.
This will setup all internal pipelines and close them afterwards automatically.

.. code-block:: python

    from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
    predictor = Predictor.from_checkpoint(
        params=PredictorParams(),
        checkpoint='PATH_TO_THE_MODEL_WITHOUT_EXT')

    for sample in predictor.predict_raw(raw_image_generator):
        inputs, prediction, meta = sample.inputs, sample.outputs, sample.meta
        # prediction is usually what you are looking for

Whereby a ``raw_image_generator`` is of type ``Iterable[np.ndarray]`` for example a list of images:

.. code-block:: python

    raw_image_generator = [np.zeros(shape=(200, 50))]


If instead the `samples (lines) are dynamically created` during the execution and the predictor shall be kept alive use the following:

.. code-block:: python

    # Create the predictor, and the raw predictor somewhere in your code
    from calamari_ocr.ocr.predict.predictor import Predictor, PredictorParams
    predictor = Predictor.from_checkpoint(
        params=PredictorParams(),
        checkpoint='PATH_TO_THE_MODEL_WITHOUT_EXT')
    raw_predictor = predictor.raw().__enter__()  # you can also wrap the following lines in a `with`-block

    # somewhere else in your code, just call the raw_predictor with a single image
    sample = raw_predictor(raw_image)  # raw_image is e.g. np.zeros(200, 50)
    inputs, prediction, meta = sample.inputs, sample.outputs, sample.meta
    # prediction is usually what you are looking for

Have a look at the `prediction tests <https://github.com/Calamari-OCR/calamari/blob/master/calamari_ocr/test/test_prediction.py>`_ for some more examples.

Multiple models (voting)
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from calamari_ocr.ocr.predict.predictor import MultiPredictor, PredictorParams
    predictor = MultiPredictor.from_paths(
        checkpoints=['CKPT1', 'CKPT2', ...],
        params=PredictorParams())

    for sample in predictor.predict_raw(raw_image_generator):
        inputs, (results, prediction), meta = sample.inputs, sample.outputs, sample.meta
        # prediction (the voted result) is usually what you are looking for

Then, apply the ``predictor`` to any data.


Prediction Object
-------------------------

Each ``prediction`` holds a full `prediction object <https://github.com/Calamari-OCR/calamari/blob/master/calamari_ocr/ocr/predict/params.py#L34>`_ which holds the actual outcome (``prediction.sentence``) but also the single character positions and probabilities (``prediction.positions``).

Extended Prediction Data
------------------------

A **full prediction** object generated from the ``Predictor`` yields a lot of information, such as **character probabilities** or **positions**.
For a full overview see the dataclass structure :ref:`here<doc.predicting:prediction object>`.
To generate those data you can either

* pass the ``--extended_prediction_data`` parameter to ``calamari-predict`` which will create ``.json`` files with the additional data written (note the huge probability matrix is not included by default), or
* call the predictor in custom python code see :ref:`Prediction<doc.predicting:Python API>`
