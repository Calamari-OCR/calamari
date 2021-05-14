Dataset Formats
===============

File Extensions
---------------

File extensions are used to map ground truth and image files (e.g., a ``.png`` and its corresponding ``.gt.txt``).
Everything after the first dot (.) in a filename is considered as the files extension.
So do not use dots to label files.

Most dataset formats provide a ``.gt_extension`` and a ``.pred_extension`` which can be used to modify the loaded and written files.

Plain files
-----------
Files provided as single images of a line (e. g. png or jpg). The ground truth is provided as plain text files in UTF-8 encoding using the same base name as the corresponding image and `.gt.txt` as extension.

Example:

.. code-block:: shell

    + train
      - 0001.png
      - 0001.gt.txt
      - 0002.png
      - 0002.gt.txt
      - ...
    + test
      - 1001.png
      - 1001.gt.txt
      - 1002.png
      - 1002.gt.txt
      - ...

Command-Line Call:

.. code-block:: shell

    calamari-train --train.images train/*.png
    calamari-predict --data.images test/*.png

PageXML
-------

Use ``--train PageXML`` to switch mode.
Provide page images as ``--train.images`` and the corresponding ``xml``-files as ``--xml_files``

Example structure:

.. code-block:: shell

    + train
      - 0001.png
      - 0001.xml
      - 0002.png
      - 0002.xml
      - ...
    + test
      - 1001.png
      - 1001.xml
      - 1002.png
      - 1002.xml
      - ...

Call:

.. code-block:: shell

    calamari-train --train PageXML --train.images train/*.png
    calamari-predict --data PageXML --data.images test/*.png

Abbyy
-----

Use ``--train Abbyy`` to switch mode.
Provide page images as ``--train.images``, the corresponding ``xml`` files must end with ``.abbyy.xml``.

Example structure:

.. code-block:: shell

    + train
      - 0001.png
      - 0001.abbyy.xml
      - 0002.png
      - 0002.abbyy.xml
      - ...
    + test
      - 1001.png
      - 1001.abbyy.xml
      - 1002.png
      - 1002.abbyy.xml
      - ...

Call:

.. code-block:: shell

    calamari-train --train Abbyy --train.images train/*.png
    calamari-predict --data Abbyy --data.images test/*.png

HDF5
----

Use ``--train Hdf5`` to switch mode.

The content of a `h5`-file is:

* ``images``: list of raw images
* ``images_dims``: the shape of the images (numpy arrays)
* ``codec``: integer mapping to decode the ``transcriptions`` (ASCII)
* ``transcripts``: list of encoded transcriptions using the codec

Call

.. code-block:: shell

    calamari-train --train Hdf5 --train.files train.h5
    calamari-predict --train Hdf5 --train.files test.h5
