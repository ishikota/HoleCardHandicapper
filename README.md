# HoleCardHandicapper
[![PyPI](https://img.shields.io/pypi/v/HoleCardHandicapper.svg?maxAge=2592000)]([![PyPI](https://img.shields.io/pypi/v/nine.svg?maxAge=2592000)](https://github.com/ishikota/HoleCardHandicapper))
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/ishikota/kyoka/blob/master/LICENSE.md)

*HoleCardHandicapper* makes a prediction of heads-up win rates on poker hands.  

```python
from holecardhandicapper.model.neuralnet import Neuralnet
from pypokerengine.engine.card import Card

model = Neuralnet("preflop")
model.compile()
holecard = [Card(suit=Card.HEART, rank=1), Card(suit=Card.SPADE, rank=1)]
model.predict(holecard) // returns 0.850103
```

You can check out sample code for all street (preflop, flop, turn, river)
- https://github.com/ishikota/HoleCardHandicapper/blob/master/sample/how_to_use.py

## Installation
`pip install HoleCardHandicapper`

## Dependency
- [Keras](https://github.com/fchollet/keras)
  - To make a prediction, *HoleCardHandicapper* internally uses *keras* library.
- [PyPokerEngine](https://github.com/ishikota/PyPokerEngine)
  - To represents trump card, we use `pypokerengine.engine.card.Card` class.
