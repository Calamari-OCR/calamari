Install
=======

Calamari requires:

* Python3.7 or later including the development packages.
* (optional) cuda/cudnn libs for GPU support, see `tensorflow <https://www.tensorflow.org/install/source#tested_build_configurations>`_ for the versions which are required/compatible.

Calamari was tested on Linux, but should be flawlessly usable on windows or mac.

See also the install instructions for `tfaip <https://tfaip.readthedocs.io/en/latest/doc.installation.html>`_ and `tensorflow <https://www.tensorflow.org/install>`_.


Installation using Pip
----------------------

The suggested method is to install calamari into a virtual environment using pip:

.. code-block:: shell

    virtualenv -p python3 PATH_TO_VENV_DIR # (e.g. virtualenv -p python3 calamari_venv)
    source PATH_TO_VENV_DIR/bin/activate
    pip install calamari-ocr

which will install Calamari and all of its dependencies.

To install the package without a virtual environment simply run

.. code-block:: shell

    pip install calamari-ocr

Installation from Source
------------------------

To install the package from its source, download the source code and install it.
Optionally (but recommended) install in a virtual env.

.. code-block:: shell

    git clone https://github.com/calamari-OCR/calamari
    cd calamari
    python setup.py install

Conda users can alternatively call

.. code-block:: shell

    conda env create -f environment_master.yml

Development Setup
-----------------

Calamari uses `black <https://black.readthedocs.io>`_ code styling.
It is recommended to integrate running black as pre-commit hook.
The following guide helps to setup everything.

The requirements ``pre-commit`` and ``black`` are already part of the requirements.
Setup ``pre-commit`` by calling:

.. code-block:: shell

    pre-commit install

To upgrade the pre-commit packages call

.. code-block:: shell

    pre-commit autoupdate
