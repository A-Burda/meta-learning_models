#IMPORT PACKAGES
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense

#PARAMETERS
n_trials = 100
epochs = 1
alpha = 0.1 #learning rate
beta = 2 #reverse temperature
w = np.random.random(3) #weights

#STATIC MODEL DATA
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).repeat(4, axis=0) #data set
test_x = np.copy(train_x)
train_y_AND = np.array([0, 0, 0, 1]).reshape(4, 1).repeat(4, axis=0)
train_y_XOR = np.array([0, 1, 1, 0]).reshape(4, 1).repeat(4, axis=0)
train_y_RM = np.random.randint(0, 2, size=(4, 1)).repeat(4, axis=0) # it is random during learning

#MODELS STRUCTURE
n_input, n_output = train_x.shape[1], 1 

model_dict = {
    'AND_model': Sequential([
        Dense(4, activation='tanh', input_shape=(n_input,)),
        Dense(n_output, activation='sigmoid')
        ]),
    'XOR_model': Sequential([
        Dense(4, activation='tanh', input_shape=(n_input,)),
        Dense(n_output, activation='sigmoid')
        ]),
    'RM_model': Sequential([
        Dense(4, activation='tanh', input_shape=(n_input,)),
        Dense(n_output, activation='sigmoid')
        ]),
    }

for model in model_dict.values(): 
    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.1), 
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy()]
    )

#MAIN LOOP 
#iniating data collection
data = pd.DataFrame()
action_list = list(range(3))
action_counts = {action: 0 for action in action_list}
previous_AND_accuracy = None
previous_XOR_accuracy = None
previous_RM_accuracy = None

# function to track the loss and accuracy of each task
def learning_track():
    test_results_AND = model_dict['AND_model'].evaluate(train_x, train_y_AND)
    data.loc[i, 'loss_AND'] = test_results_AND[0]
    data.loc[i, 'acc_AND'] = test_results_AND[1]
    
    test_results_XOR = model_dict['XOR_model'].evaluate(train_x, train_y_XOR)
    data.loc[i, 'loss_XOR'] = test_results_XOR[0]
    data.loc[i, 'acc_XOR'] = test_results_XOR[1]
    
    test_results_RM = model_dict['RM_model'].evaluate(train_x, train_y_RM)
    data.loc[i, 'loss_RM'] = test_results_RM[0]
    data.loc[i, 'acc_RM'] = test_results_RM[1]

# track the loss and acc of each task before the model training
i = 0
learning_track()

##action choice
for i in range(1, n_trials+1):
    #record the trial
    data.loc[i, 'trial'] = i
    print("trial number:", data.loc[i, 'trial'])
     
    #changeable data
    train_y_RM = np.random.randint(0, 2, size=(4, 1)).repeat(4, axis=0)
    
    #make a choice (softmax function)
    prob = np.exp(beta*w)/np.sum(np.exp(beta*w))
    data.loc[i, ['prob_RM', 'prob_XOR', 'prob_AND']] = prob
    chosen_task = np.random.choice(3, p = prob)
    
    #record the choice
    data.loc[i, 'chosen_task'] = chosen_task
    action_counts[chosen_task] += 1
    
##running the chosen task and retrieving previous accuracy
    if chosen_task == 2:
        model_name = 'AND_model'
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_AND, batch_size = 1, epochs=epochs)
        
    elif chosen_task == 1:
        model_name = 'XOR_model'
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_XOR, batch_size = 1, epochs=epochs)
        
    elif chosen_task == 0: 
        model_name = 'RM_model'
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_RM, batch_size = 1, epochs=epochs)    

## track the loss and acc of each task after training
    learning_track()
        
##learning update
    acc_model = ['acc_RM', 'acc_XOR', 'acc_AND'][chosen_task]
    previous_accuracy = data.loc[i-1, acc_model]
    trial_accuracy = data.loc[i, acc_model]
    #reward = abs(previous_accuracy - trial_accuracy) #the reward is learning progress
    reward = trial_accuracy - previous_accuracy
    data.loc[i, 'reward'] = reward
    #w[chosen_task] += alpha*(reward - w[chosen_task]) #Rescorla-Wagner learning rule
    print(data['reward'].iloc[-1])   
    
##record accuracy for the chosen task
    if chosen_task == 2: 
        previous_AND_accuracy = trial_accuracy
        
    elif chosen_task == 1: 
        previous_XOR_accuracy = trial_accuracy
      
    elif chosen_task == 0:
        previous_RM_accuracy = trial_accuracy


##GATHER DATA
##test the models
test_results_AND = model_dict['AND_model'].evaluate(test_x, train_y_AND)
test_predictions_AND = model_dict['AND_model'].predict(test_x)

test_results_XOR = model_dict['XOR_model'].evaluate(test_x, train_y_XOR)
test_predictions_XOR = model_dict['XOR_model'].predict(test_x)

test_results_RM = model_dict['RM_model'].evaluate(test_x, train_y_RM)
test_predictions_RM = model_dict['RM_model'].predict(test_x)

##summary
print(["Weights:", w])
print("Action selection frequency:", action_counts)
print("Most selected action:", max(action_counts, key=action_counts.get))
print("mean reward: {:.2f}".format(np.mean(data['reward'])))
print(f"Model name: AND Model, Loss: {test_results_AND[0]:.4f}, Accuracy: {test_results_AND[1]:.4f}")
print(f"Model name: XOR Model, Loss: {test_results_XOR[0]:.4f}, Accuracy: {test_results_XOR[1]:.4f}")
print(f"Model name: RM Model, Loss: {test_results_RM[0]:.4f}, Accuracy: {test_results_RM[1]:.4f}")

##plots
###accuracy over time 
data['mean_acc'] = data.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].mean(axis=1)

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Accuracy over Time for Each Model")
ax.plot(data['trial'], data['acc_AND'], color = "blue", label = "AND")
ax.plot(data['trial'], data['acc_XOR'], color = "green", label = "XOR")
ax.plot(data['trial'], data['acc_RM'], color = "red", label = "RM")
ax.plot(data['trial'], data['mean_acc'], color="orange", linestyle='--', label='All')
ax.set_xlabel("trial")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()

###loss over time
data['mean_loss'] = data.loc[:, ['loss_AND', 'loss_XOR', 'loss_RM']].mean(axis=1)
 
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Loss Functions for Each Model")
ax.plot(data['trial'], data['loss_AND'], color = "blue", label = "AND")
ax.plot(data['trial'], data['loss_XOR'], color = "green", label = "XOR")
ax.plot(data['trial'], data['loss_RM'], color = "red", label = "RM")
ax.plot(data['trial'], data['mean_loss'], color="orange", linestyle='--', label='All')
ax.set_xlabel("trial")
ax.set_ylabel("loss")
ax.legend()
plt.show()

###action probability over time
window_conv = 5
avg_RM = np.convolve(data['prob_RM'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_XOR = np.convolve(data['prob_XOR'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_AND = np.convolve(data['prob_AND'], np.ones(window_conv)/window_conv, mode = 'valid')
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Choice Probabilities over Time")
ax.plot(avg_RM, color='red', label = "RM")
ax.plot(avg_XOR, color='green', label = "XOR")
ax.plot(avg_AND, color='blue', label = "AND")
ax.set_xlabel("trial")
ax.set_ylabel("choice probability")
ax.legend()
plt.show()

###save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''
