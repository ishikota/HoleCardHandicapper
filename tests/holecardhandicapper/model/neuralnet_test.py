import os
from tests.base_unittest import BaseUnitTest
from holecardhandicapper.model.neuralnet import Neuralnet
from pypokerengine.engine.card import Card

class NeuralnetTest(BaseUnitTest):

  def test_predict_on_preflop(self):
    model = Neuralnet("preflop")
    model.compile()
    hole = [Card(suit=Card.HEART, rank=1), Card(suit=Card.SPADE, rank=1)]
    prediction = model.predict(hole)
    self.eq(0.862, prediction)

  def test_predict_on_flop(self):
    model = Neuralnet("flop")
    model.compile()
    hole = [Card(suit=Card.SPADE, rank=4), Card(suit=Card.SPADE, rank=2)]
    community = [Card(suit=Card.DIAMOND, rank=12), Card(suit=Card.CLUB, rank=11), Card(suit=Card.DIAMOND, rank=10)]
    prediction = model.predict(hole, community)
    self.almosteq(0.29, prediction, 0.01)

  def test_predict_on_turn(self):
    model = Neuralnet("turn")
    model.compile()
    hole = [Card(suit=Card.CLUB, rank=1), Card(suit=Card.DIAMOND, rank=10)]
    community = [Card(suit=Card.HEART, rank=6), Card(suit=Card.HEART, rank=9), Card(suit=Card.DIAMOND, rank=8), Card(suit=Card.CLUB, rank=11)]
    prediction = model.predict(hole, community)
    self.almosteq(0.50, prediction, 0.01)

  def test_predict_on_river(self):
    model = Neuralnet("river")
    model.compile()
    hole = [Card(suit=Card.SPADE, rank=1), Card(suit=Card.HEART, rank=4)]
    community = [Card(suit=Card.SPADE, rank=2), Card(suit=Card.SPADE, rank=4), Card(suit=Card.DIAMOND, rank=4), Card(suit=Card.HEART, rank=1), Card(suit=Card.HEART, rank=5)]
    prediction = model.predict(hole, community)
    self.almosteq(0.97, prediction, 0.01)


  def get_root_path(self):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..")

