#import ts and datasets
import tensorflow as tf
import numpy as np
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
        metrics = ['accuracy']
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

#MAIN LOOP
##action choice
for i in range(n_trials): 
    trial.append(i + 1) #counting the trials
    print("trial number:", trial[-1])
    prob = np.exp(beta*w)/np.sum(np.exp(beta*w)) #softmax function
    chosen_task = np.random.choice(3, p = prob)
    choice_list.append(chosen_task) 
    action_counts[chosen_task] += 1
    
##running the chosen task while tracking loss and accuracy
    if chosen_task == 2:
        model_name = 'AND_model'
        train_y_AND = np.array([0, 0, 0, 1])
        train_y_AND = train_y_AND.reshape(4, 1)
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_AND, batch_size = 1, epochs=epochs)
        loss_history_AND.append(history.history["loss"][0])
        acc_history_AND.append(history.history["accuracy"][0])
        
    elif chosen_task == 1:
        model_name = 'XOR_model'
        train_y_XOR = np.array([0, 1, 1, 0])
        train_y_XOR = train_y_XOR.reshape(4, 1)
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_XOR, batch_size = 1, epochs=epochs)
        loss_history_XOR.append(history.history["loss"][0])
        acc_history_XOR.append(history.history["accuracy"][0])
        
    elif chosen_task == 0: 
        model_name = 'RM_model'
        train_y_RM = np.random.randint(0, 2, size=(4, 1))
        model = model_dict[model_name]
        history = model.fit(train_x, train_y_RM, batch_size = 1, epochs=epochs)
        loss_history_RM.append(history.history["loss"][0])
        acc_history_RM.append(history.history["accuracy"][0])
        
##learning update
    reward = history.history["accuracy"][0] #the reward is accuracy
    reward_list.append(reward)
    w[chosen_task] += alpha*(reward - w[chosen_task]) #Rescorla-Wagner learning rule

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
print("mean reward: {:.2f}".format(np.mean(reward_list)))
print(f"Model name: AND Model, Loss: {test_results_AND[0]:.4f}, Accuracy: {test_results_AND[1]:.4f}")
print(f"Model name: XOR Model, Loss: {test_results_XOR[0]:.4f}, Accuracy: {test_results_XOR[1]:.4f}")
print(f"Model name: RM Model, Loss: {test_results_RM[0]:.4f}, Accuracy: {test_results_RM[1]:.4f}")

##plots
window_conv = 5
avg_choices = np.convolve(choice_list, np.ones(window_conv)/window_conv, mode = 'valid')
avg_reward = np.convolve(reward_list, np.ones(window_conv)/window_conv, mode = 'valid')
fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Average Choices and Reward over Time")
ax.plot(avg_reward *2, color='blue', alpha = 0.5, label = "Reward")
ax.plot(avg_choices, color='black', alpha = 0.7, label = "Choice")
ax.set_xlabel("trial")
ax.set_ylabel("choice")
ax.set_yticks(range(3))
ax.legend()
plt.show()

fig, ax = plt.subplots(1)
ax.set_title("Comparison of Loss Functions for Each Model")
ax.plot(loss_history_AND, color = "blue", label = "Easy task")
ax.plot(loss_history_XOR, color = "green", label = "Hard task")
ax.plot(loss_history_RM, color = "red", label = "Impossible task")
ax.set_xlabel("trial")
ax.set_ylabel("loss")
ax.legend()
plt.show()

###save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''


