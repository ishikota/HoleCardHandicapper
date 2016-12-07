from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten
from holecardhandicapper.model.const import PREFLOP, FLOP, TURN, RIVER
from holecardhandicapper.model.utils import Utils

class ModelBuilder:

  @classmethod
  def build_model(self, street):
    if PREFLOP == street:
      return self.build_preflop_cache_model()
    elif FLOP == street:
      return self.build_flop_model()
    elif TURN == street:
      return self.build_turn_model()
    elif RIVER == street:
      return self.build_river_model()

  @classmethod
  def build_preflop_cache_model(self):
      return CacheModel()

  @classmethod
  def build_preflop_model(self):
    model = Sequential()
    model.add(Dense(256, input_dim=52))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(64))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss="mse",  optimizer="adam")
    return model

  @classmethod
  def build_flop_model(self):
    model = Sequential()

    model.add(Convolution2D(128, 3, 3, border_mode="same", input_shape=(1, 5, 17)))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(128, 3, 3, border_mode="same", input_shape=(1, 5, 17)))
    model.add(Activation("relu"))
    model.add(Convolution2D(128, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss="mse",  optimizer="adadelta")
    return model

  @classmethod
  def build_turn_model(self):
    model = Sequential()

    model.add(Convolution2D(256, 3, 3, border_mode="same", input_shape=(1, 6, 17)))
    model.add(Activation("relu"))
    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, border_mode="same", input_shape=(1, 6, 17)))
    model.add(Activation("relu"))
    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss="mse",  optimizer="adadelta")
    return model

  @classmethod
  def build_river_model(self):
    model = Sequential()

    model.add(Convolution2D(256, 3, 3, border_mode="same", input_shape=(1, 7, 17)))
    model.add(Activation("relu"))
    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(Convolution2D(256, 3, 3, border_mode="same"))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss="mse",  optimizer="adadelta")
    return model

class CacheModel(object):

  def predict(self, X):
    ids = [i for i in range(len(X[0])) if X[0][i]==1]
    ids = [i+1 for i in ids]
    win_rate = self.table[ids[0]][ids[1]]
    return [[win_rate]]

  def load_weights(self, _filepath):
    self.table = Utils.build_preflop_winrate_table()

