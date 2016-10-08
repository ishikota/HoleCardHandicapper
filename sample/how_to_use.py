import os
import sys
root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root)
sys.path.append(os.path.join(root, "holecardhandicapper"))

from holecardhandicapper.model.neuralnet import Neuralnet
from pypokerengine.engine.card import Card

# preflop
preflop_model = Neuralnet("preflop")
preflop_model.compile()
holecard = [Card(suit=Card.HEART, rank=1), Card(suit=Card.SPADE, rank=1)]
prediction = preflop_model.predict(holecard)
print "win rate of [ holecard=%s ] is %f" % (map(str, holecard), prediction)

# flop
flop_model = Neuralnet("flop")
flop_model.compile()
holecard = [Card(suit=Card.SPADE, rank=4), Card(suit=Card.SPADE, rank=2)]
communitycard = [Card(suit=Card.DIAMOND, rank=12), Card(suit=Card.CLUB, rank=11), Card(suit=Card.DIAMOND, rank=10)]
prediction = flop_model.predict(holecard, communitycard)
print "win rate of [ holecard=%s, communitycard=%s ] is %f" % (map(str, holecard), map(str, communitycard), prediction)

# turn
turn_model = Neuralnet("turn")
turn_model.compile()
holecard = [Card(suit=Card.CLUB, rank=1), Card(suit=Card.DIAMOND, rank=10)]
communitycard = [Card(suit=Card.HEART, rank=6), Card(suit=Card.HEART, rank=9), Card(suit=Card.DIAMOND, rank=8), Card(suit=Card.CLUB, rank=11)]
prediction = turn_model.predict(holecard, communitycard)
print "win rate of [ holecard=%s, communitycard=%s ] is %f" % (map(str, holecard), map(str, communitycard), prediction)

# river
river_model = Neuralnet("river")
river_model.compile()
holecard = [Card(suit=Card.SPADE, rank=1), Card(suit=Card.HEART, rank=4)]
communitycard = [Card(suit=Card.SPADE, rank=2), Card(suit=Card.SPADE, rank=4), Card(suit=Card.DIAMOND, rank=4), Card(suit=Card.HEART, rank=1), Card(suit=Card.HEART, rank=5)]
prediction = river_model.predict(holecard, communitycard)
print "win rate of [ holecard=%s, communitycard=%s ] is %f" % (map(str, holecard), map(str, communitycard), prediction)
