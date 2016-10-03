
# coding: utf-8

import os
import time
root = os.path.join(os.path.dirname(__file__), "..", "..")
script_dir = os.path.join(root, "notebook", "script")
timestamp = time.time()
print "timestamp of this script is %d" % timestamp
gen_save_file_path = lambda suffix: os.path.join(script_dir, "turn_cnn_training_note_%d.%s" % (timestamp, suffix))

# # Import learning data

# In[22]:

import pandas as pd

training_data_path = root + "/learning_data/data/win_rate/turn/50000-data-1000-simulation-2-players-win-rate-data.csv"
test_data_path = root + "/learning_data/data/win_rate/turn/10000-data-1000-simulation-2-players-win-rate-data.csv"
train_df = pd.read_csv(training_data_path)
test_df = pd.read_csv(test_data_path)


# # About learning data

# In[23]:

print train_df.shape, test_df.shape


# In[24]:

train_df.head()


# In[25]:

train_df.describe()


# In[26]:

test_df.describe()


# # Data Processing

# ## card id -> 1-hot vector

# In[28]:

import numpy as np
from pypokerengine.engine.card import Card

fetch_hole = lambda row: [row[key] for key in ['hole1_id', 'hole2_id']]
fetch_community = lambda row: [row[key] for key in ['community1_id', 'community2_id', 'community3_id', 'community4_id']]
gen_card_rank_vec = lambda card: [1 if r == card.rank else 0 for r in range(2,15)]
gen_card_suit_vec = lambda card: [1 if s == card.suit else 0 for s in [Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE]]
gen_card_vec = lambda card: gen_card_rank_vec(card) + gen_card_suit_vec(card)
gen_img = lambda zipped: [gen_card_vec(Card.from_id(card_id)) for card_id in zipped[0]+zipped[1]]

train_hole = train_df.apply(lambda row: fetch_hole(row), axis=1)
train_community = train_df.apply(lambda row: fetch_community(row), axis=1)
train_df["onehot"] = map(gen_img, zip(train_hole, train_community))

test_hole = test_df.apply(lambda row: fetch_hole(row), axis=1)
test_community = test_df.apply(lambda row: fetch_community(row), axis=1)
test_df["onehot"] = map(gen_img, zip(test_hole, test_community))


# ## Format data (pandas.df -> numpy.ndarray)

# In[29]:

wrap = lambda lst: np.array(lst)
to_ndarray = lambda X: wrap([wrap(x) for x in X])
train_x, train_y = [to_ndarray(array) for array in [train_df["onehot"].values, train_df["win_rate"].values]]
test_x, test_y = [to_ndarray(array) for array in [test_df["onehot"].values, test_df["win_rate"].values]]
train_x, test_x = [X.reshape(X.shape[0], 1, X.shape[1], X.shape[2]) for X in [train_x, test_x]]
print "shape of training data => x: %s, y: %s" % (train_x.shape, train_y.shape)
print "shape of test data => x: %s, y: %s" % (test_x.shape, test_y.shape)


# # Create model

# In[30]:

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Activation, Dropout, Flatten

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


# # Train model

# In[31]:

history = model.fit(train_x, train_y, batch_size=128, nb_epoch=500, validation_split=0.1, verbose=2)

import json
with open(gen_save_file_path("json"), "wb") as f:
  json.dump(model.to_json(), f)
model.save_weights(gen_save_file_path("h5"))

# # Check model performance

# ## Visualize loss transition

# In[32]:

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn

plt.figure(figsize=(10,10))
for idx, key in enumerate(history.history, start=1):
    plt.subplot(2, 1, idx)
    plt.plot(range(len(history.history[key])), history.history[key])
    plt.title(key)
plt.savefig(gen_save_file_path("png"))


# ## Test model performance by MSE 

# In[33]:

from sklearn.metrics import mean_squared_error

def print_model_performance(model, train_x, train_y, test_x, test_y):
    print 'MSE on training data = {score}'.format(score=mean_squared_error(model.predict(train_x), train_y))
    print 'MSE on test data = {score}'.format(score=mean_squared_error(model.predict(test_x), test_y))


# In[34]:

print_model_performance(model, train_x, train_y, test_x, test_y)


# ## See model prediction on sample data

# In[35]:

from pypokerengine.engine.card import Card
C, D, H, S = Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE

test_case = [
    [(8, D), (4, C), (13, C), (1, C), (6, C), (2, C), 0.994],
    [(3, D), (3, H), (5, H), (4, D), (5, S), (4, H), 0.11],
    [(1, C), (10, D), (6, H), (9, H), (8, D), (11, C), 0.505]
]

to_id = lambda card: card.to_id()
gen_card_rank_vec = lambda card: [1 if r == card.rank else 0 for r in range(2,15)]
gen_card_suit_vec = lambda card: [1 if s == card.suit else 0 for s in [Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE]]
gen_card_vec = lambda card: gen_card_rank_vec(card) + gen_card_suit_vec(card)
gen_img = lambda zipped: [gen_card_vec(Card.from_id(card_id)) for card_id in zipped[0]+zipped[1]]
wrap = lambda lst: np.array(lst)
to_ndarray = lambda X: wrap([wrap(x) for x in X])

for card1, card2, card3, card4, card5, card6, expected in test_case:
    cards = [Card(rank=rank, suit=suit) for rank, suit in [card1, card2, card3, card4, card5, card6]]
    hole = cards[:2]
    community = cards[2:]
    hole_ids = map(to_id, hole)
    community_ids = map(to_id, community)
    x = gen_img((hole_ids, community_ids))
    X = to_ndarray(x)
    X = np.array([X.reshape(1, X.shape[0], X.shape[1])])
    y = model.predict(X)[0][0]
    print "HOLE = [%s, %s], COMMUNITY = [%s, %s, %s, %s] => win_rate = { prediction=%f, expected=%f }" % tuple(map(str, hole) + map(str, community) + [y , expected])

