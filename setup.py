from setuptools import setup

setup(
    name = 'HoleCardHandicapper',
    version = '0.0.1',
    description = 'NeuralNet for poker hands evaluation ',
    license = 'MIT',
    author = 'ishikota',
    author_email = 'ishikota086@gmail.com',
    url = 'https://github.com/ishikota/HoleCardHandicapper',
    keywords = 'python poker hand predict evaluate',
    packages = ['holecardhandicapper', 'holecardhandicapper.model'],
    install_requires = ['Keras==1.0.4', 'h5py==2.6.0', 'PyPokerEngine==0.0.1'],
    dependency_links = ['git+https://github.com/ishikota/PyPokerEngine.git@0.0.1#egg=PyPokerEngine-0.0.1']
    )
