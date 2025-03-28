#IMPORT PACKAGES
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense

#PARAMETERS
n_trials = 300
model_runs = 15 
epochs = 1
alpha = 0.3 #learning rate
beta = 2 #reverse temperature
w = np.array([1/3, 1/3, 1/3])
#np.random.random(3) #weights

#X DATA
##data set
x_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

##RM context
context_vector_RM = np.array([0, 0, 1])
context_vector_RM = np.tile(context_vector_RM, (4, 1))

train_x_RM = np.hstack([x_inputs, context_vector_RM]) 
test_x_RM = np.copy(train_x_RM)

##XOR context
context_vector_XOR = np.array([0, 1, 0])
context_vector_XOR = np.tile(context_vector_XOR, (4, 1))

train_x_XOR = np.hstack([x_inputs, context_vector_XOR]) 
test_x_XOR = np.copy(train_x_XOR)

##AND context
context_vector_AND = np.array([1, 0, 0])
context_vector_AND = np.tile(context_vector_AND, (4, 1))

train_x_AND = np.hstack([x_inputs, context_vector_AND]) 
test_x_AND = np.copy(train_x_AND)

#STATIC Y DATA
train_y_AND = np.array([0, 0, 0, 1]).reshape(4, 1)
train_y_XOR = np.array([0, 1, 1, 0]).reshape(4, 1)

#FUCTIONS 
#random Y data for RM
def random_y_data(): 
    return np.random.randint(0, 2, size=(4, 1))

##tracking loss and accuracy
def learning_track():
    test_results_RM = model.evaluate(train_x_RM, train_y_RM)
    data.loc[index, 'loss_RM'] = test_results_RM[0]
    data.loc[index, 'acc_RM'] = test_results_RM[1]

    test_results_XOR = model.evaluate(train_x_XOR, train_y_XOR)
    data.loc[index, 'loss_XOR'] = test_results_XOR[0]
    data.loc[index, 'acc_XOR'] = test_results_XOR[1]
        
    test_results_AND = model.evaluate(train_x_AND, train_y_AND)
    data.loc[index, 'loss_AND'] = test_results_AND[0]
    data.loc[index, 'acc_AND'] = test_results_AND[1]

#FULL TRAINING LOOP
#initialising loop data collection
data = pd.DataFrame()
loop = 0
index = 0

for run in range(model_runs):
    
    #initialising/reseting trial data collection 
    trial = 0
    loop += 1
    action_list = list(range(3))
    action_counts = {action: 0 for action in action_list}
    
    #MODEL STRUCTURE
    model = Sequential([
        Dense(8, activation='tanh', input_shape=(5,)),
        Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05), 
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy()]
        )
       
    #record data before training
    train_y_RM = random_y_data()

    #TRIAL LOOP 
    ##action choice
    for i in range(n_trials):
        #record the trial and run
        index += 1
        trial += 1
        data.loc[index, 'loop'] = loop
        data.loc[index, 'trial'] = trial
        print("trial number:", data.loc[index, 'trial'])
         
        #changeable y_data
        train_y_RM = random_y_data()
        
        #make a choice (softmax function)
        prob = np.exp(beta*w)/np.sum(np.exp(beta*w))
        data.loc[index, ['prob_RM', 'prob_XOR', 'prob_AND']] = prob
        chosen_task = np.random.choice(3, p = prob)
        
        #record the choice
        data.loc[index, 'chosen_task'] = chosen_task 
        action_counts[chosen_task] += 1
        
    ##set the context
        if chosen_task == 0:
            task_name = 'RM_task'
            history = model.fit(train_x_RM, train_y_RM, batch_size = 1, epochs=epochs)

        elif chosen_task == 1: 
            task_name = 'XOR_task'
            history = model.fit(train_x_XOR, train_y_XOR, batch_size = 1, epochs=epochs)
                  
        elif chosen_task == 2:
            task_name = 'AND_task'
            history = model.fit(train_x_AND, train_y_AND, batch_size = 1, epochs=epochs)
        
    ##record data after training
        learning_track()
              
    ##averaging previous accuracies
        if chosen_task == 0:
            trial_acc = data['acc_RM'].iloc[-1]

        elif chosen_task == 1: 
            trial_acc = data['acc_XOR'].iloc[-1]
                  
        elif chosen_task == 2:
            trial_acc = data['acc_AND'].iloc[-1]

        reward = trial_acc #reward is accuracy

    ##learning update
        data.loc[index, 'reward'] = reward
        w[chosen_task] += alpha*(reward - w[chosen_task]) #Rescorla-Wagner learning rule
        print(data['reward'].iloc[-1]) 
        
    ##GATHER DATA
    ##test the models
    test_results_AND = model.evaluate(test_x_AND, train_y_AND)
    test_predictions_AND = model.predict(test_x_AND)

    test_results_XOR = model.evaluate(test_x_XOR, train_y_XOR)
    test_predictions_XOR = model.predict(test_x_XOR)

    test_results_RM = model.evaluate(test_x_RM, train_y_RM)
    test_predictions_RM = model.predict(test_x_RM)
    
    ##summary
    print(["Weights:", w])
    print("Action selection frequency:", action_counts)
    print("Most selected action:", max(action_counts, key=action_counts.get))
    print("mean reward: {:.2f}".format(np.mean(data['reward'])))
    print(f"Task name: AND, Loss: {test_results_AND[0]:.4f}, Accuracy: {test_results_AND[1]:.4f}")
    print(f"Task name: XOR, Loss: {test_results_XOR[0]:.4f}, Accuracy: {test_results_XOR[1]:.4f}")
    print(f"Task name: RM, Loss: {test_results_RM[0]:.4f}, Accuracy: {test_results_RM[1]:.4f}")
    
##plots
window_conv = 5

###get mean and std
data_fuse = data.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].mean()
data_std = data.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].std()

###action probability over time
con_RM_prob = np.convolve(data_fuse['prob_RM'], np.ones(window_conv)/window_conv, mode = 'valid')
con_XOR_prob = np.convolve(data_fuse['prob_XOR'], np.ones(window_conv)/window_conv, mode = 'valid')
con_AND_prob = np.convolve(data_fuse['prob_AND'], np.ones(window_conv)/window_conv, mode = 'valid')
std_RM_prob = np.convolve(data_std['prob_RM'], np.ones(window_conv)/window_conv, mode='valid')
std_XOR_prob = np.convolve(data_std['prob_XOR'], np.ones(window_conv)/window_conv, mode='valid')
std_AND_prob = np.convolve(data_std['prob_AND'], np.ones(window_conv)/window_conv, mode='valid')
trials = np.arange(len(con_RM_prob))

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Choice Probabilities over Time")
ax.plot(con_RM_prob, color='red', label = "RM")
ax.fill_between(trials, con_RM_prob - 1.96 * std_RM_prob, con_RM_prob + 1.96 * std_RM_prob, color='red', alpha=0.2)
ax.plot(con_XOR_prob, color='green', label = "XOR")
ax.fill_between(trials, con_XOR_prob - 1.96 * std_XOR_prob, con_XOR_prob + 1.96 * std_XOR_prob, color='green', alpha=0.2)
ax.plot(con_AND_prob, color='blue', label = "AND")
ax.fill_between(trials, con_AND_prob - 1.96 * std_AND_prob, con_AND_prob + 1.96 * std_AND_prob, color='blue', alpha=0.2)
ax.set_xlabel("trial")
ax.set_ylabel("choice probability")
ax.legend()
plt.show()
plt.close(fig)

###accuracy over time 
data_fuse['mean_acc'] = data_fuse.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].mean(axis=1)

avg_RM_acc = np.convolve(data_fuse['acc_RM'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_XOR_acc = np.convolve(data_fuse['acc_XOR'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_AND_acc = np.convolve(data_fuse['acc_AND'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_mean_acc = np.convolve(data_fuse['mean_acc'], np.ones(window_conv)/window_conv, mode = 'valid')
std_RM_acc = np.convolve(data_std['acc_RM'], np.ones(window_conv)/window_conv, mode='valid')
std_XOR_acc = np.convolve(data_std['acc_XOR'], np.ones(window_conv)/window_conv, mode='valid')
std_AND_acc = np.convolve(data_std['acc_AND'], np.ones(window_conv)/window_conv, mode='valid')

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Accuracy over Time for Each Task")
ax.plot(avg_AND_acc, color = "blue", label = "AND")
ax.fill_between(trials, avg_AND_acc - 1.96 * std_AND_acc, avg_AND_acc + 1.96 * std_AND_acc, color='blue', alpha=0.2)
ax.plot(avg_XOR_acc, color = "green", label = "XOR")
ax.fill_between(trials, avg_XOR_acc - 1.96 * std_XOR_acc, avg_XOR_acc + 1.96 * std_XOR_acc, color='green', alpha=0.2)
ax.plot(avg_RM_acc, color = "red", label = "RM")
ax.fill_between(trials, avg_RM_acc - 1.96 * std_RM_acc, avg_RM_acc + 1.96 * std_RM_acc, color='red', alpha=0.1)
ax.plot(avg_mean_acc, color="orange", linestyle='--', label='All')
ax.set_xlabel("trial")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()
plt.close(fig)

###loss over time
data_fuse['mean_loss'] = data_fuse.loc[:, ['loss_AND', 'loss_XOR', 'loss_RM']].mean(axis=1)
 
avg_RM_loss = np.convolve(data_fuse['loss_RM'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_XOR_loss = np.convolve(data_fuse['loss_XOR'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_AND_loss = np.convolve(data_fuse['loss_AND'], np.ones(window_conv)/window_conv, mode = 'valid')
avg_mean_loss = np.convolve(data_fuse['mean_loss'], np.ones(window_conv)/window_conv, mode = 'valid')
std_RM_loss = np.convolve(data_std['loss_RM'], np.ones(window_conv)/window_conv, mode='valid')
std_XOR_loss = np.convolve(data_std['loss_XOR'], np.ones(window_conv)/window_conv, mode='valid')
std_AND_loss = np.convolve(data_std['loss_AND'], np.ones(window_conv)/window_conv, mode='valid')

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Loss Functions for Each Task")
ax.plot(avg_AND_loss, color = "blue", label = "AND")
ax.fill_between(trials, avg_AND_loss - 1.96 * std_AND_loss, avg_AND_loss + 1.96 * std_AND_loss, color='blue', alpha=0.2)
ax.plot(avg_XOR_loss, color = "green", label = "XOR")
ax.fill_between(trials, avg_XOR_loss - 1.96 * std_XOR_loss, avg_XOR_loss + 1.96 * std_XOR_loss, color='green', alpha=0.2)
ax.plot(avg_RM_loss, color = "red", label = "RM")
ax.fill_between(trials, avg_RM_loss - 1.96 * std_RM_loss, avg_RM_loss + 1.96 * std_RM_loss, color='red', alpha=0.1)
ax.plot(avg_mean_loss, color="orange", linestyle='--', label='All')
ax.set_xlabel("trial")
ax.set_ylabel("loss")
ax.legend()
plt.show()
plt.close(fig)

###save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''