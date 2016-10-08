import os
from holecardhandicapper.model.const import PREFLOP, FLOP, TURN, RIVER

class Utils:

  @classmethod
  def get_model_weights_path(self, street):
    if PREFLOP == street:
      return self.get_preflop_model_weights_path()
    if FLOP == street:
      return self.get_flop_model_weights_path()
    if TURN == street:
      return self.get_turn_model_weights_path()
    if RIVER == street:
      return self.get_river_model_weights_path()
    raise ValueError("Unknown street [ %s ] received." % street)


  @classmethod
  def get_preflop_model_weights_path(self):
    root = self.__get_root_path()
    return os.path.join(root, "notebook", "mlp", "resource", "preflop_neuralnet_model_weights.h5")

  @classmethod
  def get_flop_model_weights_path(self):
    root = self.__get_root_path()
    return os.path.join(root, "notebook", "cnn", "resource", "flop_cnn_model_weights.h5")

  @classmethod
  def get_turn_model_weights_path(self):
    root = self.__get_root_path()
    return os.path.join(root, "notebook", "cnn", "resource", "turn_cnn_model_weights.h5")

  @classmethod
  def get_river_model_weights_path(self):
    root = self.__get_root_path()
    return os.path.join(root, "notebook", "cnn", "resource", "river_cnn_model_weights.h5")

  @classmethod
  def __get_root_path(self):
    return os.path.join(os.path.dirname(__file__), "..", "..")

