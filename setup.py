from setuptools import setup, find_packages

setup(
    name='calamari',
    version='0.1.0',
    packages=find_packages(),
    license='GPL-v3.0',
    long_description=open("README.md").read(),
    include_package_data=True,
    author="Christoph Wick",
    author_email="christoph.wick@informatik.uni-wuerzburg.de",
    url="http://github.com/chwick/calamari",
    entry_points={
        'console_scripts': [
            'calamari-eval=calamari.scripts.eval:main',
            'calamari-predict=calamari.scripts.predict:main',
            'calamari-resume-training=calamari.scripts.resume_training:main',
            'calamari-train=calamari.scripts.train:main',
        ],
    },
    install_requires=open("requirements.txt").read().split(),
)
