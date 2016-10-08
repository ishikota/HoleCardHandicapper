import numpy as np
import json
from pypokerengine.engine.card import Card
from holecardhandicapper.model.model_builder import ModelBuilder
from holecardhandicapper.model.utils import Utils
from holecardhandicapper.model.const import PREFLOP, FLOP, TURN, RIVER

class Neuralnet(object):

  def __init__(self, street):
    self.model = None
    self.street = street

  def compile(self):
    self.model = ModelBuilder.build_model(self.street)
    self.model.load_weights(Utils.get_model_weights_path(self.street))

  def predict(self, hole, community=[]):
    X = self.__generate_input_vector(self.street, hole, community)
    return self.model.predict(X)[0][0]

  def __read_json_from_file(self, filepath):
    with open(filepath, "rb") as f:
      return json.load(f)

  def __generate_input_vector(self, street, hole, community):
    if street == PREFLOP:
      return self.__generate_input_for_mlp(hole, community)
    elif street in [FLOP, TURN, RIVER]:
      return self.__generate_input_for_cnn(hole, community)
    else:
      raise ValueError("Unknown street [ %s ] is received" % street)

  def __generate_input_for_mlp(self, hole, community):
    gen_one_hot = lambda target_ids: [1 if i in target_ids else 0 for i in range(1,53)]
    to_id = lambda card: card.to_id()
    x = gen_one_hot(map(to_id, hole+community))
    X = np.array([x])
    return X

  def __generate_input_for_cnn(self, hole, community):
   to_id = lambda card: card.to_id()
   gen_card_rank_vec = lambda card: [1 if r == card.rank else 0 for r in range(2,15)]
   gen_card_suit_vec = lambda card: [1 if s == card.suit else 0 for s in [Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE]]
   gen_card_vec = lambda card: gen_card_rank_vec(card) + gen_card_suit_vec(card)
   gen_img = lambda zipped: [gen_card_vec(Card.from_id(card_id)) for card_id in zipped[0]+zipped[1]]
   wrap = lambda lst: np.array(lst)
   to_ndarray = lambda X: wrap([wrap(x) for x in X])
   hole_ids = map(to_id, hole)
   community_ids = map(to_id, community)
   x = gen_img((hole_ids, community_ids))
   X = to_ndarray(x)
   X = np.array([X.reshape(1, X.shape[0], X.shape[1])])
   return X

