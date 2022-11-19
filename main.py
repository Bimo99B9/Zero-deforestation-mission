# %% [markdown]
# # Zero deforestation mission

# %% [markdown]
# ## Labels:
# 
# **0**: "Plantation". Network of rectangular plantation blocks.
# 
# **1**: "Grassland/Shrubland". Large homogeneous areas with few trees.
# 
# **2**: "Smallholder Agriculture": Small scale area, in wich you can find deforestation covered by agriculture.

# %%
# Importing modules 
import numpy as np 
import pandas as pd 
import os
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout, BatchNormalization
from tensorflow.keras.models import Sequential


# from keras.utils import to_categorical
# from keras.layers import Dense,Conv2D,Flatten,MaxPool2D,Dropout, BatchNormalization
# from keras.models import Sequential

from sklearn.model_selection import train_test_split

np.random.seed(1)

# %%
df_train = pd.read_csv("train.csv")
df_test = pd.read_csv("test.csv")


# %%
def calc_ndvi(image):
    b, g, r = cv2.split(image)
    bottom = (r.astype(float) + b.astype(float))
    bottom[bottom==0] = 0.01
    ndvi = (b.astype(float) - r) / bottom
    return ndvi

# %%
# Process training data.
train_images = []
train_path = df_train["example_path"].to_numpy()

for filename in train_path:
    img = cv2.imread(filename)
    #img = cv2.filter2D(src=img, ddepth=-1, kernel=calc_ndvi(img))
    train_images.append(img)

train_images = np.array(train_images)
train_images.shape

# %%
# Process test data.
test_images = []
test_path = df_test["example_path"].to_numpy()

for filename in test_path:
    img = cv2.imread(filename)
    #img = cv2.filter2D(src=img, ddepth=-1, kernel=calc_ndvi(img))
    test_images.append(img)

test_images = np.array(test_images)
test_images.shape

# %% [markdown]
# Here we remove all the attributes for the data CSVs, so we only work with the label. (Currently).
# 
# We also convert 0 to [1 0 0], 1 to [0 1 0], and 2 to [0 0 1]. Therefore, we can translate the ML algorithm into a multiple binary classification problem.

# %%
df_train_label = df_train.pop("label")
df_train_label = pd.get_dummies(df_train_label).values

# %%
X_train, X_val, y_train, y_val = train_test_split(train_images, df_train_label, random_state=1234)

# %% [markdown]
# So now, we have:
# 
# **X_train**: Images for training. (TRAINING)
# 
# **X_val**: Images for validating the model. (VALIDATION)
# 
# 
# **y_train**: Labels of the training images, so we can train. (TRAINING)
# 
# **y_val**: Labels of the validation set, so we can compare the results we predicted over X_val. (VALIDATION - F1 SCORE)
# 
# 
# **test_images**: Like X_train or X_val, but for the final answer of the challenge (We don't know the labels).

# %%
# X_train = X_train / 255
# X_val = X_val / 255
# display(X_train.min())
# display(X_train.max())

# %%
#Let's plot one of our examples.
print(df_train_label[5])
plt.imshow(train_images[5])

# %%
# Let's see a different one.
print(df_train_label[100])
plt.imshow(train_images[100])

# %%
# Sequential model.

model = Sequential()
model.add(Conv2D(kernel_size=(3,3), filters=75, activation='tanh', input_shape=(332,332,3,)))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))
model.add(MaxPool2D(2,2))
model.add(Conv2D(filters=30,kernel_size = (3,3),activation='tanh'))

model.add(Flatten())

model.add(Dense(20,activation='relu'))
model.add(Dense(15,activation='relu'))
model.add(Dense(3,activation = 'softmax'))
    
model.compile(
              loss='categorical_crossentropy', 
              metrics=['acc'],
              optimizer='adam'
             )

# %%
# Sequential model.

# num_classes = 3

# model = Sequential()
# model.add(Conv2D(75, (3, 3), strides=1, padding="same", activation="relu", input_shape=(332,332,3,)))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(50, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(Dropout(0.2))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Conv2D(25, (3, 3), strides=1, padding="same", activation="relu"))
# model.add(BatchNormalization())
# model.add(MaxPool2D((2, 2), strides=2, padding="same"))
# model.add(Flatten())
# model.add(Dense(units=512, activation="relu"))
# model.add(Dropout(0.3))
# model.add(Dense(units=num_classes, activation="softmax"))
    
# model.compile(
#               loss='categorical_crossentropy', 
#               metrics=['acc']
#               #optimizer='adam'
#              )

# %%
model.summary()

# %%
# Training the model.
history = model.fit(X_train, y_train, epochs=50, batch_size=50, validation_data=(X_val, y_val))
#history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# %%
# summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()


