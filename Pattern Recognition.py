import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import csv
from sklearn.model_selection import train_test_split

'''
with open("liteshadow4-packets/original/EmergeSync.cap", "rb") as f:
    data = f.read()
'''

column_names = ["No.", "Time", "Source", "Destination", "Proto", "Len", "Info"]
attributes = ["Device Type", "File Type", "File Size", "Transfer Attempts", "Connection Interval", "Repeat Count"]

#train_dataset_fp = tf.keras.utils.get_file(fname=)
#train_dataset = tf.data.experimental.make_csv_dataset(csvfile, batch_size, column_names=column_names,
#                                                      label_name=attributes, num_epochs=7)

batch_size = 1



csv = "EmergeSync.csv"
val_csv = "EmergeSync.csv"

dataframe = pd.read_csv(csv, sep="\t")

print(dataframe)

train, test_ds = train_test_split(dataframe, train_size=0.8, test_size=0.2)
train_ds, val_ds = train_test_split(train, train_size=0.8, test_size=0.2)

print(len(train_ds), 'train examples')
print(len(val_ds), 'validation examples')
print(len(test_ds), 'test examples')


#with open("EmergeSync.csv", newline='') as csvfile: # // Already opened in train dataset
#    spamreader = csv.reader(csvfile, delimiter='\n', quotechar='|')

#train_dataset = [0, 2, 3, 8, 12, 24]
#validation_dataset = [0, 2, 3, 8, 12, 24]

#train_dataset = tf.data.experimental.make_csv_dataset(csv, batch_size)
#validation_dataset = tf.data.experimental.make_csv_dataset(val_csv, batch_size)

#train_dataset = tf.data.TFRecordDataset(csv)
#validation_dataset = tf.data.TFRecordDataset(val_csv)
    #for row in spamreader:
        #print(','.join(row))



def load_data(input_data):
    list_key = []
    list_value = []
    for key in input_data:
        list_key.append(key)
        list_value.append(input_data[key])
    return list_key, list_value


#train_dataset = tf.data.experimental.make_csv_dataset(spamreader, 2, column_names=column_names, label_name=attributes, num_epochs=7)

#model = keras.Sequential([keras.layers.Dense(units=7, activation="linear", input_dim=2),
#                         keras.layers.Dense(7, activation="linear"),
#                         keras.layers.Dense(7, activation="linear")])

model = keras.Sequential()

units = 5

model.add(keras.layers.Flatten(input_shape=(583, 1)))
model.add(keras.layers.Dense(units=units, activation='linear', input_dim=units))
model.add(keras.layers.Dense(units=units, activation='linear'))
model.add(keras.layers.Dense(units=units, activation='linear'))
print(model)

train_array = np.asarray(train_ds)
val_array = np.asarray(val_ds)

model.compile(optimizer="adam", loss="mean_squared_error", metrics=["accuracy"])

num_epochs = 2

history = model.fit(x=train_array, epochs=num_epochs, steps_per_epoch=5, shuffle=True, validation_data=val_array, verbose=1)
print(history)

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

