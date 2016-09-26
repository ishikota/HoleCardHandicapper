from pypokerengine.engine.card import Card
from pypokerengine.engine.hand_evaluator import HandEvaluator
from learning_data.generator.round_simulator import RoundSimulator
import random
import math

class HoleEvaluator:

  @classmethod
  def estimate_win_rate(self, hole, base_community_card, player_num, simulation_num):
    csimulation = lambda community: RoundSimulator.simulation(player_num, hole, community)
    cfillblank = lambda community: self.__fill_blank_card(hole, community)
    simulation_result = [csimulation(cfillblank(base_community_card)) for _ in range(simulation_num)]
    win_count = simulation_result.count(True)
    win_rate = 1.0 * win_count / simulation_num
    return win_rate

  @classmethod
  def draw_unknown_card(self, draw_num, known_cards):
    unknown_card_id = [cardid for cardid in range(1, 53)\
            if cardid not in [card.to_id() for card in known_cards]]
    return [Card.from_id(cardid) for cardid in random.sample(unknown_card_id, draw_num)]

  @classmethod
  def __fill_blank_card(self, holecard, community_):
    community = [Card.from_id(card.to_id()) for card in community_]  # deep copy
    need_card = 5 - len(community)
    for card in self.draw_unknown_card(need_card, holecard + community):
      community.append(card)
    return community

