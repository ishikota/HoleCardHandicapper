
# coding: utf-8

import os
import time
root = os.path.join(os.path.dirname(__file__), "..", "..")
script_dir = os.path.join(root, "notebook", "script")
timestamp = time.time()
print "timestamp of this script is %d" % timestamp
gen_save_file_path = lambda suffix: os.path.join(script_dir, "preflop_neuralnet_training_note_%d.%s" % (timestamp, suffix))


# # Import learning data

# In[164]:

import pandas as pd

training_data_path = root + "/learning_data/data/win_rate/preflop/50000-data-1000-simulation-2-players-win-rate-data.csv"
test_data_path = root + "/learning_data/data/win_rate/preflop/10000-data-1000-simulation-2-players-win-rate-data.csv"
train_df = pd.read_csv(training_data_path)
test_df = pd.read_csv(test_data_path)


# # About learning data

# In[165]:

print train_df.shape, test_df.shape


# In[166]:

train_df.head()


# In[159]:

train_df.describe()


# In[160]:

test_df.describe()


# # Data Processing

# ## card id -> 1-hot vector

# In[167]:

import numpy as np

gen_one_hot = lambda h1, h2: [1 if i in [h1, h2] else 0 for i in range(1,53)]
train_df["onehot"] = train_df.apply(lambda row: gen_one_hot(row['hole1_id'], row['hole2_id']), axis=1)
test_df["onehot"] = test_df.apply(lambda row: gen_one_hot(row['hole1_id'], row['hole2_id']), axis=1)


# ## Format data (pandas.df -> numpy.ndarray)

# In[180]:

to_ndarray = lambda X: np.array([np.array(x) for x in X])
train_x, train_y = [to_ndarray(array) for array in [train_df["onehot"].values, train_df["win_rate"].values]]
test_x, test_y = [to_ndarray(array) for array in [test_df["onehot"].values, test_df["win_rate"].values]]
print "shape of training data => x: %s, y: %s" % (train_x.shape, train_y.shape)
print "shape of test data => x: %s, y: %s" % (test_x.shape, test_y.shape)


# # Create model

# In[171]:

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout

model = Sequential()
model.add(Dense(256, input_dim=52))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation("relu"))
model.add(Dropout(0.5))
model.add(Dense(1))
model.compile(loss="mse",  optimizer="adam")


# # Train model

# In[172]:

history = model.fit(train_x, train_y, batch_size=128, nb_epoch=2000, validation_split=0.1, verbose=1)

import json
with open(gen_save_file_path("json"), "wb") as f:
  json.dump(model.to_json(), f)
model.save_weights(gen_save_file_path("h5"))


# # Check model performance

# ## Visualize loss transition

# In[174]:
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

# In[178]:

from sklearn.metrics import mean_squared_error

def print_model_performance(model, train_x, train_y, test_x, test_y):
    print 'MSE on training data = {score}'.format(score=mean_squared_error(model.predict(train_x), train_y))
    print 'MSE on test data = {score}'.format(score=mean_squared_error(model.predict(test_x), test_y))


# In[179]:

print_model_performance(model, train_x, train_y, test_x, test_y)


# ## See model prediction on sample data

# In[177]:

from pypokerengine.engine.card import Card

test_case = [
    (1, Card.SPADE, 1, Card.CLUB, 0.871), 
    (3, Card.HEART, 2, Card.SPADE, 0.477),
    (10, Card.CLUB, 13, Card.HEART, 0.676)
]
gen_one_hot = lambda h1, h2: [1 if i in [h1, h2] else 0 for i in range(1,53)]

df = pd.DataFrame()
for rank1, suit1, rank2, suit2, expected in test_case:
    hole1 = Card(rank=rank1, suit=suit1)
    hole2 = Card(rank=rank2, suit=suit2)
    hole = [hole1, hole2]
    x = gen_one_hot(*[card.to_id() for card in hole])
    X = np.array([x])
    y = model.predict(X)[0][0]
    print "HOLE = [%s, %s] => win_rate = { prediction=%f, expected=%f }" % tuple(map(str, hole)+ [y , expected])

