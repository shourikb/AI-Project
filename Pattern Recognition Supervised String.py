import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import csv
import os
from sklearn.model_selection import train_test_split

'''
with open("liteshadow4-packets/original/EmergeSync.cap", "rb") as f:
    data = f.read()
'''
#csv = "MyCapTest.csv"
#val_csv = "MyCapTest.csv"

train_dataset_url = "file:///C:/Users/Shourik/PycharmProjects/AI/EmergeSyncString.csv"

train_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(train_dataset_url),
                                           origin=train_dataset_url)

test_dataset_url = "file:///C:/Users/Shourik/PycharmProjects/AI/EmergeSyncStringTest.csv"

test_dataset_fp = tf.keras.utils.get_file(fname=os.path.basename(test_dataset_url),
                                           origin=test_dataset_url)


print("Local copy of the dataset file: {}".format(train_dataset_fp))

#column_names = ["No.", "Time", "Source", "Destination", "Proto", "Len", "Info", "Anomaly"]
column_names = ["No.", "Time", "Source", "Destination", "Proto", "Anomaly"]

class_names = ["anomaly", "normal"]

batch_size = 1

#dataframe = pd.read_csv(csv)

print("Train Dataset FP: ")
print(pd.read_csv(train_dataset_fp))

feature_names = column_names[:-1]
label_name = column_names[-1]

train_dataset = tf.data.experimental.make_csv_dataset(
    train_dataset_fp,
    batch_size=batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1)

print(train_dataset)

features, labels = next(iter(train_dataset))

print(features)

def pack_features_vector(features, labels):
  """Pack the features into a single array."""
  features = tf.stack(list(features.values()), axis=1)
  return features, labels

train_dataset = train_dataset.map(pack_features_vector)

features, labels = next(iter(train_dataset))

print("Features")
print(features[:5])

model = tf.keras.Sequential([
  tf.keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(5,)),  # input shape required
  tf.keras.layers.Dense(10, activation=tf.nn.relu),
  tf.keras.layers.Dense(2)
])

predictions = model(features)
print("Predictions")
print(predictions[:5])

tf.nn.softmax(predictions[:5])

print("Prediction: {}".format(tf.argmax(predictions, axis=1)))
print("    Labels: {}".format(labels))

loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

def loss(model, x, y, training):
  # training=training is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  y_ = model(x, training=training)

  return loss_object(y_true=y, y_pred=y_)


l = loss(model, features, labels, training=False)
print("Loss test: {}".format(l))


def grad(model, inputs, targets):
  with tf.GradientTape() as tape:
    loss_value = loss(model, inputs, targets, training=True)
  return loss_value, tape.gradient(loss_value, model.trainable_variables)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

loss_value, grads = grad(model, features, labels)

print("Step: {}, Initial Loss: {}".format(optimizer.iterations.numpy(),
                                          loss_value.numpy()))

optimizer.apply_gradients(zip(grads, model.trainable_variables))

print("Step: {},         Loss: {}".format(optimizer.iterations.numpy(),
                                          loss(model, features, labels, training=True).numpy()))

## Note: Rerunning this cell uses the same model variables

# Keep results for plotting
train_loss_results = []
train_accuracy_results = []

num_epochs = 11

for epoch in range(num_epochs):
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

  # Training loop - using batches of 32
  for x, y in train_dataset:
    # Optimize the model
    loss_value, grads = grad(model, x, y)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    # Track progress
    epoch_loss_avg.update_state(loss_value)  # Add current batch loss
    # Compare predicted label to actual label
    # training=True is needed only if there are layers with different
    # behavior during training versus inference (e.g. Dropout).
    epoch_accuracy.update_state(y, model(x, training=True))

  # End epoch
  train_loss_results.append(epoch_loss_avg.result())
  train_accuracy_results.append(epoch_accuracy.result())

  if epoch % 5 == 0:
    print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                epoch_loss_avg.result(),
                                                                epoch_accuracy.result()))

test_dataset = tf.data.experimental.make_csv_dataset(
    test_dataset_fp,
    batch_size,
    column_names=column_names,
    label_name=label_name,
    num_epochs=1,
    shuffle=False)

test_dataset = test_dataset.map(pack_features_vector)

test_accuracy = tf.keras.metrics.Accuracy()

for (x, y) in test_dataset:
  # training=False is needed only if there are layers with different
  # behavior during training versus inference (e.g. Dropout).
  logits = model(x, training=False)
  prediction = tf.argmax(logits, axis=1, output_type=tf.int32)
  test_accuracy(prediction, y)

print("Test set accuracy: {:.3%}".format(test_accuracy.result()))

print(tf.stack([y,prediction],axis=1))

predict_dataset = tf.convert_to_tensor([
    ["5s", "0.3s", "192.120.10", "132.011.20", "RSYNC"],
    ["6s", "0.5s", "124.150.01", "132.---.13", "--.Nc"],
    ["7s", "0.3s", "132.230.12", "192.120.10", "RSYNC"]
])

# training=False is needed only if there are layers with different
# behavior during training versus inference (e.g. Dropout).
predictions = model(predict_dataset, training=False)

for i, logits in enumerate(predictions):
  class_idx = tf.argmax(logits).numpy()
  p = tf.nn.softmax(logits)[class_idx]
  name = class_names[class_idx]
  print("Example {} prediction: {} ({:4.1f}%)".format(i, name, 100*p))
'''
'''
train, test_ds = train_test_split(dataframe, train_size=0.8, test_size=0.2)
train_ds, val_ds = train_test_split(train, train_size=0.8, test_size=0.2)

print(len(train_ds), 'train examples')
print(len(val_ds), 'validation examples')
print(len(test_ds), 'test examples')

def df_to_dataset(dataframe, shuffle=True, batch_size=32):
  dataframe = dataframe.copy()
  labels = dataframe.pop('Anomaly')
  ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
  if shuffle:
    ds = ds.shuffle(buffer_size=len(dataframe))
  ds = ds.batch(batch_size)
  return ds

train_dataset = df_to_dataset(train_ds, batch_size=batch_size)
test_ds = df_to_dataset(test_ds, shuffle=False, batch_size=batch_size)
val_ds = df_to_dataset(test_ds, shuffle=False, batch_size=batch_size)

print(train_dataset)



#print("Features: {}".format(feature_names))
#print("Label: {}".format(label_name))



#train_dataset = tf.data.experimental.make_csv_dataset(spamreader, 2, column_names=column_names, label_name=attributes, num_epochs=7)

#model = keras.Sequential([keras.layers.Dense(units=7, activation="linear", input_dim=2),
#                         keras.layers.Dense(7, activation="linear"),
#                         keras.layers.Dense(7, activation="linear")])
'''
model = keras.Sequential()

units = 7

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(units=units, activation='linear'))
model.add(keras.layers.Dense(units=units, activation='linear'))
model.add(keras.layers.Dense(units=units, activation='linear'))
print(model)



model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

num_epochs = 2

history = model.fit(train_ds, epochs=num_epochs, steps_per_epoch=5, batch_size=batch_size, shuffle=True, validation_data=val_ds, verbose=1)
print(history)
'''
#model.fit(train_dataset, column_names, epochs=7)


#data = {0:0, 8:0, 2:0, 6:0, 11:1, 35:1, 3:0, 5:0, 7:0, 30:1}

#(train_characters, train_labels), (test_characters, test_labels) = data

#random.shuffle(data)

#train_packets = data[:75]
#test_packets = data[:25]


'''
(train_chars, train_labels) = load_data()
(test_chars, test_labels) = load_data()

print(train_chars)
print(train_labels)
print(test_chars)
print(test_labels)

#class_names = ["Digits", "Alphabet"]


h = keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(1,))

print(h)

z =  keras.layers.Dense(10, activation=tf.nn.relu)

print(z)

c = keras.layers.Dense(2)
print(c)

model = keras.Sequential([keras.layers.Dense(10, activation=tf.nn.relu, input_shape=(1,)),
                         keras.layers.Dense(10, activation=tf.nn.relu),
                         keras.layers.Dense(6)])
print(model)


#loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])

model.fit(train_chars, train_labels, epochs=7)

prediction = model.predict(test_chars)

for i in range(3):
    print("Actual: " + class_names[test_labels[i]])
    print("Prediction: " + class_names[np.argmax(prediction[i])])
'''

