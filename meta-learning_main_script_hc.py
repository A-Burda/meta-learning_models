#import ts and datasets
import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense

#CODE STARTS HERE
#PARAMETERS
n_trials = 100
epochs = 1
alpha = 0.1 #learning rate
beta = 2 #reverse temperature
w = np.random.random(3) #weights
train_x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]) #data set
test_x = np.copy(train_x)

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
        metrics = [keras.metrics.BinaryAccuracy()]
    )

#MAIN LOOP 
#iniating lists
trial = []
reward_list = [] 
choice_list = []
loss_history_AND = []
loss_history_XOR = []
loss_history_RM = []
acc_history_AND = []
acc_history_XOR = []
acc_history_RM = []
action_list = list(range(3))
action_counts = {action: 0 for action in action_list}

# data
data = pd.DataFrame()

#MAIN LOOP
##action choice
for i in range(n_trials): 
    trial.append(i + 1) #counting the trials
    print("trial number:", trial[-1])
    prob = np.exp(beta*w)/np.sum(np.exp(beta*w)) #softmax function
    chosen_task = np.random.choice(3, p = prob)
    choice_list.append(chosen_task) 
    action_counts[chosen_task] += 1
    
    ## record the trial
    data.loc[i, 'trial'] = i + 1   
    ## record the choice
    data.loc[i, 'chosen_task'] = chosen_task
    
    ## y label of each task
    # AND
    train_y_AND = np.array([0, 0, 0, 1])
    train_y_AND = train_y_AND.reshape(4, 1)
    # XOR
    train_y_XOR = np.array([0, 1, 1, 0])
    train_y_XOR = train_y_XOR.reshape(4, 1)
    # RM
    train_y_RM = np.random.randint(0, 2, size=(4, 1))
    
    ##running the chosen task while tracking loss and accuracy
    if chosen_task == 2:
        model_name = 'AND_model'
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_AND, batch_size = 1, epochs=epochs)
        loss_history_AND.append(history.history["loss"][0])
        acc_history_AND.append(history.history["binary_accuracy"][0])
    
    elif chosen_task == 1:
        model_name = 'XOR_model'
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_XOR, batch_size = 1, epochs=epochs)
        loss_history_XOR.append(history.history["loss"][0])
        acc_history_XOR.append(history.history["binary_accuracy"][0])
        
    elif chosen_task == 0: 
        model_name = 'RM_model'
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_RM, batch_size = 1, epochs=epochs)
        loss_history_RM.append(history.history["loss"][0])
        acc_history_RM.append(history.history["binary_accuracy"][0])
    
    ## record the loss and acc of each task after training
    # And task
    test_results_AND = model_dict['AND_model'].evaluate(train_x, train_y_AND)
    data.loc[i, 'loss_and'] = test_results_AND[0]
    data.loc[i, 'acc_and'] = test_results_AND[1]
    
    # XOR task
    test_results_XOR = model_dict['XOR_model'].evaluate(train_x, train_y_XOR)
    data.loc[i, 'loss_xor'] = test_results_XOR[0]
    data.loc[i, 'acc_xor'] = test_results_XOR[1]
    
    # RM task
    test_results_RM = model_dict['RM_model'].evaluate(train_x, train_y_RM)
    data.loc[i, 'loss_rm'] = test_results_RM[0]
    data.loc[i, 'acc_rm'] = test_results_RM[1]
    
##learning update
    reward = history.history["binary_accuracy"][0] #the reward is accuracy
    reward_list.append(reward)
    w[chosen_task] += alpha*(reward - w[chosen_task]) #Rescorla-Wagner learning rule

##GATHER DATA
##test the models
test_results_AND = model_dict['AND_model'].evaluate(test_x, train_y_AND, batch_size=1)
test_predictions_AND = model_dict['AND_model'].predict(test_x)

test_results_XOR = model_dict['XOR_model'].evaluate(test_x, train_y_XOR)
test_predictions_XOR = model_dict['XOR_model'].predict(test_x)

test_results_RM = model_dict['RM_model'].evaluate(test_x, train_y_RM)
test_predictions_RM = model_dict['RM_model'].predict(test_x)

##summary
print(["Weights:", w])
print("Action selection frequency:", action_counts)
print("Most selected action:", max(action_counts, key=action_counts.get))
print("mean reward: {:.2f}".format(np.mean(reward_list)))
print(f"Model name: AND Model, Loss: {test_results_AND[0]:.4f}, Accuracy: {test_results_AND[1]:.4f}")
print(f"Model name: XOR Model, Loss: {test_results_XOR[0]:.4f}, Accuracy: {test_results_XOR[1]:.4f}")
print(f"Model name: RM Model, Loss: {test_results_RM[0]:.4f}, Accuracy: {test_results_RM[1]:.4f}")

#%% plot loss or acc?
y_value = 'acc'

## mean loss and acc across all task
data['mean_loss'] = data.loc[:, ['loss_and', 'loss_xor', 'loss_rm']].mean(axis=1)
data['mean_acc'] = data.loc[:, ['acc_and', 'acc_xor', 'acc_rm']].mean(axis=1)

## plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title(f"Comparison of {y_value} for Each Model")
# loss functions of each task
if y_value == 'loss':
    ax.plot(data['trial'], data['loss_and'], color = "blue", label = "AND")
    ax.plot(data['trial'], data['loss_xor'], color = "green", label = "XOR")
    ax.plot(data['trial'], data['loss_rm'], color = "red", label = "RM")
    ax.plot(data['trial'], data['mean_loss'], color="orange", label='All')
# accuracy of each task
elif y_value == 'acc':
    ax.plot(data['trial'], data['acc_and'], color = "blue", linestyle='--', label = "AND")
    ax.plot(data['trial'], data['acc_xor'], color = "green", linestyle='--', label = "XOR")
    ax.plot(data['trial'], data['acc_rm'], color = "red", linestyle='--', label = "RM")
    ax.plot(data['trial'], data['mean_acc'], color="orange", linestyle='--', label='All')
    
# task choice in each trial
data_rm = data.loc[data['chosen_task']==0, 'trial']
ax.scatter(data_rm, np.ones_like(data_rm)*1.5, color='red')
data_xor = data.loc[data['chosen_task']==1, 'trial']
ax.scatter(data_xor, np.ones_like(data_xor)*1.5, color='green')
data_rm = data.loc[data['chosen_task']==2, 'trial']
ax.scatter(data_rm, np.ones_like(data_rm)*1.5, color='blue')
# labels
ax.set_xlabel("trial")
ax.set_ylabel(f"{y_value}")
ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.5], ['0', '0.2', '0.4', '0.6', '0.8', '1.0', 'Choice'])
ax.legend()
plt.show()

# save figure
fig.savefig(f'task_choice_and_{y_value}.pdf', dpi=300)

###save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''


