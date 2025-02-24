#import ts and datasets
import tensorflow as tf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt 

###TO DO LIST
#LVL 1
##create the tasks: AND, XOR, RM (random mapping)
##generalise the model structure to all tasks
##export final data in a ending summary
##compare performances with each other (statistics and plot)
##Level 1 is done

###CODE STARTS HERE
#initialise
learning_rate = 0.5
epochs = 100

#define the data set 
##AND function
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_x = np.copy(train_x)
train_y = np.array([0, 0, 0, 1])
train_y = train_y.reshape(4, 1)

##XOR function
'''train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
test_x = np.copy(train_x)
train_y = np.array([0, 1, 1, 0])
train_y = train_y.reshape(4, 1)'''

##Randomised function
'''to be done'''

#input/output shape
n_input, n_output = train_x.shape[1], 1 

#learning model structure
model = tf.keras.models.Sequential([
    tf.keras.Input(shape=(n_input,)),
    tf.keras.layers.Dense(n_output, activation="sigmoid")
    ])
model.build() 

#train & test the model
opt = tf.keras.optimizers.Adam(learning_rate = learning_rate)
model.compile(optimizer = opt, loss=tf.keras.losses.MeanSquaredError())
history = model.fit(train_x, train_y, batch_size = 1, epochs = epochs)
model.summary()

test_data = model.predict(test_x)

#report data
print(model.get_weights())
fig, ax = plt.subplots(1)
ax.plot(history.history["loss"], color = "black")

#test data
print("predictions on the test data:")
print(test_data)

#save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''



###ADDITIONAL CODE FOR MY OWN USAGE
'''def f(x): 
    return (x - 1)**2

def df(x):
    return 2 * (x-1)

x = 5 
learning_rate = 0.1
tolerance = 1e-6

history = [x]

for i in range(100):
    gradient = df(x)
    x = x - learning_rate * gradient
    history.append(x)
    if abs(gradient) < tolerance: 
        break

print(f"Minimum found at x = {x}, f(x) = {f(x)}")'''


