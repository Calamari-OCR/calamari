from setuptools import setup, find_packages

from calamari_ocr import __version__

setup(
    name='calamari_ocr',
    version=__version__,
    packages=find_packages(),
    license='Apache License 2.0',
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    include_package_data=True,
    author="Christoph Wick",
    author_email="christoph.wick@informatik.uni-wuerzburg.de",
    url="https://github.com/Calamari-OCR/calamari",
    download_url='https://github.com/Calamari-OCR/calamari/archive/v{}.tar.gz'.format(__version__),
    entry_points={
        'console_scripts': [
            'calamari-eval=calamari_ocr.scripts.eval:main',
            'calamari-predict=calamari_ocr.scripts.predict:main',
            'calamari-resume-training=calamari_ocr.scripts.resume_training:main',
            'calamari-train=calamari_ocr.scripts.train:main',
            'calamari-cross-fold-train=calamari_ocr.scripts.cross_fold_train:main',
            'calamari-predict-and-eval=calamari_ocr.scripts.predict_and_eval:main',
            'calamari-tensorflow-rename-variables=calamari_ocr.scripts.tensorflow_rename_variables:main',
        ],
    },
    python_requires='>=3.7',
    install_requires=open("requirements.txt").read().split('\n'),
    keywords=['OCR', 'optical character recognition', 'ocropy', 'ocropus', 'kraken'],
    data_files=[('', ["requirements.txt"])],
)
