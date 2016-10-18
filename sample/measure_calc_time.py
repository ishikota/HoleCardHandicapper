import os
import sys
root = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(root)
sys.path.append(os.path.join(root, "holecardhandicapper"))

import time

from holecardhandicapper.model.neuralnet import Neuralnet
from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck

def gen_card_set(community_num):
  deck = Deck()
  deck.shuffle()
  return deck.draw_cards(2), deck.draw_cards(5)[:community_num]

def measure_calc_time(model, community_num):
  hole, community = gen_card_set(community_num)
  start_time = time.time()
  model.predict(hole, community)
  return time.time() - start_time

TEST_NUM = 10
avg = lambda lst: 1.0 * sum(lst) / len(lst)
measure_avg_calc_time = lambda model, community_num: avg([measure_calc_time(model, community_num) for _ in range(TEST_NUM)])

# preflop
preflop_model = Neuralnet("preflop")
preflop_model.compile()
print "[preflop] Average calculation time : %f" % measure_avg_calc_time(preflop_model, 0)

# flop
flop_model = Neuralnet("flop")
flop_model.compile()
print "[flop] Average calculation time : %f" % measure_avg_calc_time(flop_model, 3)

# turn
turn_model = Neuralnet("turn")
turn_model.compile()
print "[turn] Average calculation time : %f" % measure_avg_calc_time(turn_model, 4)

# river
river_model = Neuralnet("river")
river_model.compile()
print "[river] Average calculation time : %f" % measure_avg_calc_time(river_model, 5)

