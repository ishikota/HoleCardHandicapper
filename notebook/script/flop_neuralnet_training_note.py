
# coding: utf-8

import os
import time
root = os.path.join(os.path.dirname(__file__), "..", "..")
script_dir = os.path.join(root, "notebook", "script")
timestamp = time.time()
print "timestamp of this script is %d" % timestamp
gen_save_file_path = lambda suffix: os.path.join(script_dir, "flop_neuralnet_training_note_%d.%s" % (timestamp, suffix))

# # Import learning data

# In[57]:

import pandas as pd

training_data_path = root + "/learning_data/data/win_rate/flop/50000-data-1000-simulation-2-players-win-rate-data.csv"
test_data_path = root + "/learning_data/data/win_rate/flop/10000-data-1000-simulation-2-players-win-rate-data.csv"
train_df = pd.read_csv(training_data_path)
test_df = pd.read_csv(test_data_path)


# # About learning data

# In[58]:

print train_df.shape, test_df.shape


# In[60]:

train_df.head()


# In[61]:

train_df.describe()


# In[62]:

test_df.describe()


# # Data Processing

# ## card id -> 1-hot vector

# In[63]:

import numpy as np

gen_one_hot = lambda target_ids: [1 if i in target_ids else 0 for i in range(1,53)]
fetch_hole = lambda row: [row[key] for key in ['hole1_id', 'hole2_id']]
fetch_community = lambda row: [row[key] for key in ['community1_id', 'community2_id', 'community3_id']]

train_hole_one_hot = train_df.apply(lambda row: gen_one_hot(fetch_hole(row)), axis=1)
train_community_one_hot = train_df.apply(lambda row: gen_one_hot(fetch_community(row)), axis=1)
train_df["onehot"] = train_hole_one_hot + train_community_one_hot

test_hole_one_hot = test_df.apply(lambda row: gen_one_hot(fetch_hole(row)), axis=1)
test_community_one_hot = test_df.apply(lambda row: gen_one_hot(fetch_community(row)), axis=1)
test_df["onehot"] = test_hole_one_hot + test_community_one_hot


# ## Format data (pandas.df -> numpy.ndarray)

# In[65]:

to_ndarray = lambda X: np.array([np.array(x) for x in X])
train_x, train_y = [to_ndarray(array) for array in [train_df["onehot"].values, train_df["win_rate"].values]]
test_x, test_y = [to_ndarray(array) for array in [test_df["onehot"].values, test_df["win_rate"].values]]
print "shape of training data => x: %s, y: %s" % (train_x.shape, train_y.shape)
print "shape of test data => x: %s, y: %s" % (test_x.shape, test_y.shape)


# # Create model

# In[66]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(256, input_dim=104))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss="mse",  optimizer="adam")


# # Train model

# In[67]:

history = model.fit(train_x, train_y, batch_size=128, nb_epoch=2000, validation_split=0.1, verbose=2)

import json
with open(gen_save_file_path("json"), "wb") as f:
  json.dump(model.to_json(), f)
model.save_weights(gen_save_file_path("h5"))

# # Check model performance

# ## Visualize loss transition

# In[68]:

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

# In[69]:

from sklearn.metrics import mean_squared_error

def print_model_performance(model, train_x, train_y, test_x, test_y):
    print 'MSE on training data = {score}'.format(score=mean_squared_error(model.predict(train_x), train_y))
    print 'MSE on test data = {score}'.format(score=mean_squared_error(model.predict(test_x), test_y))


# In[70]:

print_model_performance(model, train_x, train_y, test_x, test_y)


# ## See model prediction on sample data

# In[71]:

from pypokerengine.engine.card import Card
C, D, H, S = Card.CLUB, Card.DIAMOND, Card.HEART, Card.SPADE

test_case = [
    [(4, C), (10, S), (10, D), (1, C), (10, H), 0.954],
    [(4, S), (2, S), (12, D), (11, C), (10, D), 0.274],
    [(1, H), (11, H), (8, D), (3, C), (7, D), 0.507]
]

gen_one_hot = lambda target_ids: [1 if i in target_ids else 0 for i in range(1,53)]
to_id = lambda card: card.to_id()

for card1, card2, card3, card4, card5, expected in test_case:
    cards = [Card(rank=rank, suit=suit) for rank, suit in [card1, card2, card3, card4, card5]]
    hole = cards[:2]
    community = cards[2:]
    hole_onehot = gen_one_hot(map(to_id, hole))
    community_onehot = gen_one_hot(map(to_id, community))
    x = hole_onehot + community_onehot
    X = np.array([x])
    y = model.predict(X)[0][0]
    print "HOLE = [%s, %s], COMMUNITY = [%s, %s, %s] => win_rate = { prediction=%f, expected=%f }" % tuple(map(str, hole) + map(str, community) + [y , expected])

