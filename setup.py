from setuptools import setup

setup(
    name = 'HoleCardHandicapper',
    version = '1.2.0',
    author = 'ishikota',
    author_email = 'ishikota086@gmail.com',
    description = 'NeuralNet for poker hands evaluation ',
    license = 'MIT',
    keywords = 'python poker hand predict evaluate evaluator',
    url = 'https://github.com/ishikota/HoleCardHandicapper',
    packages = ['holecardhandicapper', 'holecardhandicapper.model', ],
    package_data = {'holecardhandicapper' : ['model/weights/*.h5', 'model/data/preflop_winrate.csv']},
    install_requires = ['Keras==1.0.4', 'h5py==2.6.0', 'PyPokerEngine'],
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
    ],
)
