#import ts and datasets
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation

#TO DO LIST
##LVL 1
###export final data in a ending summary
###compare performances with each other (statistics and plot)
###Level 1 is done

#CODE STARTS HERE
##DATA SET
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_x = np.copy(train_x)


##MODEL STRUCTURE
n_input, n_output = train_x.shape[1], 1 

model = Sequential()
model.add(Dense(
    units=8, activation='tanh', input_shape=(n_input,)))
model.add(Dense(
    n_output, activation='sigmoid'))
model.build()

model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1), 
    loss = tf.keras.losses.BinaryCrossentropy(),
    metrics = ['accuracy'])


##Y_DATA SET + FIT
epochs = 150

###AND function
'''model_name = 'AND_model'
train_y = np.array([0, 0, 0, 1])
train_y = train_y.reshape(4, 1)
history = model.fit(train_x, train_y, batch_size = 1, epochs=epochs)
loss_history = history.history["loss"]'''

###XOR function
model_name = 'XOR_model'
train_y = np.array([0, 1, 1, 0])
train_y = train_y.reshape(4, 1)
history = model.fit(train_x, train_y, batch_size = 1, epochs=epochs)
loss_history = history.history["loss"]

###RM loop
'''model_name = 'RM_model'
loss_history = []
accuracy_history = []

for epoch in range(epochs):
    train_y = np.random.randint(0, 2, size=(4, 1))
    print(f"Epoch {epoch+1}/{epochs}")
    history = model.fit(train_x, train_y, batch_size = 1, epochs=1, verbose=1)
    loss_history.append(history.history["loss"][0]) 
    accuracy_history.append(history.history["accuracy"][0])'''


##GATHER DATA
###store data
test_results = model.evaluate(test_x, train_y)
test_predictions = model.predict(test_x)
model.summary()

###report data: summary
print(model.get_weights())
print("predictions on the test data:")
print(test_predictions)
print(f"Model name: {model_name}, Loss: {test_results[0]:.4f}, Accuracy: {test_results[1]:.4f}")

###report data: plots
fig, ax = plt.subplots(1)
ax.plot(loss_history, color = "black")

###save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''


