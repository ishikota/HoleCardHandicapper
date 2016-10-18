from setuptools import setup

setup(
    name = 'HoleCardHandicapper',
    version = '1.0.0',
    author = 'ishikota',
    author_email = 'ishikota086@gmail.com',
    description = 'NeuralNet for poker hands evaluation ',
    license = 'MIT',
    keywords = 'python poker hand predict evaluate evaluator',
    url = 'https://github.com/ishikota/HoleCardHandicapper',
    packages = ['holecardhandicapper', 'holecardhandicapper.model'],
    install_requires = ['Keras==1.0.4', 'h5py==2.6.0', 'PyPokerEngine==0.0.1'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
