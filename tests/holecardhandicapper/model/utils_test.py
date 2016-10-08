import os
from tests.base_unittest import BaseUnitTest
from holecardhandicapper.model.utils import Utils

class UtilsTest(BaseUnitTest):

  def test_get_preflop_model_path(self):
    expected = os.path.normpath(os.path.join(self.get_root(), "notebook", "mlp", "resource", "preflop_neuralnet_model_weights.h5"))
    self.eq(expected, os.path.normpath(Utils.get_model_weights_path("preflop")))

  def test_get_flop_model_path(self):
    expected = os.path.normpath(os.path.join(self.get_root(), "notebook", "cnn", "resource", "flop_cnn_model_weights.h5"))
    self.eq(expected, os.path.normpath(Utils.get_model_weights_path("flop")))

  def test_get_turn_model_path(self):
    expected = os.path.normpath(os.path.join(self.get_root(), "notebook", "cnn", "resource", "turn_cnn_model_weights.h5"))
    self.eq(expected, os.path.normpath(Utils.get_model_weights_path("turn")))

  def test_get_river_model_path(self):
    expected = os.path.normpath(os.path.join(self.get_root(), "notebook", "cnn", "resource", "river_cnn_model_weights.h5"))
    self.eq(expected, os.path.normpath(Utils.get_model_weights_path("river")))

  def test_get_root(self):
    self.eq("HoleCardHandicapper", os.path.basename(os.path.normpath(self.get_root())))

  def get_root(self):
    return os.path.join(os.path.dirname(__file__), "..", "..", "..")
