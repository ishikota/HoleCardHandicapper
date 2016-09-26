from pypokerengine.engine.card import Card
from pypokerengine.engine.deck import Deck
from pypokerengine.engine.hand_evaluator import HandEvaluator

class RoundSimulator:

  @classmethod
  def simulation(self, player_num, holecard, community_card):
    deck = self.__setup_deck(holecard + community_card)
    my_score = HandEvaluator.eval_hand(holecard, community_card)
    others_hole = [deck.draw_cards(2) for _ in range(player_num-1)]
    others_score = [HandEvaluator.eval_hand(hole, community_card) for hole in others_hole]
    return my_score >= max(others_score)

  @classmethod
  def __setup_deck(self, used_cards):
    deck = Deck()
    deck = self.__extract_used_card(deck, used_cards)
    deck.shuffle()
    return deck

  @classmethod
  def __extract_used_card(self, deck, used_cards):
    deck.deck = [Card.from_id(card.to_id()) for card in deck.deck \
        if not card.to_id() in [used_card.to_id() for used_card in used_cards]]
    return deck

