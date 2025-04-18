#IMPORT PACKAGES
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
from keras.models import Sequential
from keras.layers.core import Dense

#PARAMETERS
tested_model = 'model_LVL2_LP_signed'
n_trials = 300
model_runs = 15 
epochs = 1
alpha = 0.3 #learning rate
w = np.array([1/3, 1/3, 1/3])
#np.random.random(3) #weights
window_size = 20 

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

#RUN MODEL FUNCTION
def model_train_loop(): 
    
    #FUNCTIONS
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
    
    #initialising loop data collection
    data = pd.DataFrame()
    loop = 0
    index = 0

    #reverse temperature (and creating the random sampling data model)
    if tested_model == 'random_sampling_model': 
        beta = 0 

    else: 
        beta = 2
         
    #FULL TRAINING LOOP
    for run in range(model_runs):
        
        #initialising/reseting trial data collection 
        trial = 0
        loop += 1
        action_list = list(range(3))
        action_counts = {action: 0 for action in action_list}
        loss_history = {0: [], 1: [], 2: []}
        
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
            
            #set the context
            if chosen_task == 0:
                task_name = 'RM_task'
                history = model.fit(train_x_RM, train_y_RM, batch_size = 1, epochs=epochs)

            elif chosen_task == 1: 
                task_name = 'XOR_task'
                history = model.fit(train_x_XOR, train_y_XOR, batch_size = 1, epochs=epochs)
                      
            elif chosen_task == 2:
                task_name = 'AND_task'
                history = model.fit(train_x_AND, train_y_AND, batch_size = 1, epochs=epochs)
            
            #record data after training
            learning_track()
            
            #averaging previous loss
            if chosen_task == 0:
                trial_loss = data['loss_RM'].iloc[-1]

            elif chosen_task == 1: 
                trial_loss = data['loss_XOR'].iloc[-1]
                          
            elif chosen_task == 2:
                trial_loss = data['loss_AND'].iloc[-1]

            loss_history[chosen_task].append(trial_loss)
            
            ##Choosing the reward signal based on the model        
            if tested_model == 'model_LVL2_LP_signed': 
                if len(loss_history[chosen_task]) > window_size:
                    loss_history[chosen_task].pop(0)
                           
                if len(loss_history[chosen_task]) > 1:
                    reward = -(np.mean(np.diff(loss_history[chosen_task])))  #reward is the mean progress
                else:
                    reward = 0

            if tested_model == 'model_LVL2_unsigned':
                if len(loss_history[chosen_task]) > window_size:
                    loss_history[chosen_task].pop(0)
                           
                if len(loss_history[chosen_task]) > 1:
                    reward = -(np.mean(np.diff(loss_history[chosen_task])))  #reward is the mean progress
                else:
                    reward = 0
                    
            if tested_model == 'model_LVL2_acc': 
                reward = -(trial_loss) #reward is accuracy
             
        #if the task does random sampling, reward doesn't matter
            if tested_model == 'random_sampling_model' 
                reward = 0         

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
  
    return data
    

###run and compare models
data_1 = model_train_loop()
tested_model = 'random_sampling_model'
data_2 = model_train_loop()

#plot parameters
window_conv = 5
confidence_level = 1.96

#shaping the data
data_fuse_1 = data_1.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].mean()
data_std_1 = data_1.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].std()

data_fuse_2 = data_2.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].mean()
data_std_2 = data_2.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].std()

data_fuse_1['mean_acc'] = data_fuse_1.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].mean(axis=1)
data_fuse_1['std_acc'] = data_fuse_1.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].std(axis=1)

data_fuse_2['mean_acc'] = data_fuse_2.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].mean(axis=1)
data_fuse_2['std_acc'] = data_fuse_2.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].std(axis=1)

#AVG: compare accuracy over time
avg_mean_acc_1 = np.convolve(data_fuse_1['mean_acc'], np.ones(window_conv)/window_conv, mode = 'valid')
std_mean_acc_1 = np.convolve(data_fuse_1['std_acc'], np.ones(window_conv)/window_conv, mode='valid')
avg_mean_acc_2 = np.convolve(data_fuse_2['mean_acc'], np.ones(window_conv)/window_conv, mode = 'valid')
std_mean_acc_2 = np.convolve(data_fuse_2['std_acc'], np.ones(window_conv)/window_conv, mode='valid')
trials = np.arange(len(avg_mean_acc_1))

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Accuracy over Time Across Tasks for Each Model")
ax.plot(avg_mean_acc_1, color="blue", label='our model')
ax.fill_between(trials, avg_mean_acc_1 - 1.96 * std_mean_acc_1, avg_mean_acc_1 + 1.96 * std_mean_acc_1, color='blue', alpha=0.1)
ax.plot(avg_mean_acc_2, color="grey", label='random sampling')
ax.fill_between(trials, avg_mean_acc_2 - 1.96 * std_mean_acc_2, avg_mean_acc_2 + 1.96 * std_mean_acc_2, color='grey', alpha=0.1)
ax.set_xlabel("trial")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()
plt.close(fig)

#AND: compare accuracy over time
avg_AND_acc_1 = np.convolve(data_fuse_1['acc_AND'], np.ones(window_conv)/window_conv, mode = 'valid')
std_AND_acc_1 = np.convolve(data_std_1['acc_AND'], np.ones(window_conv)/window_conv, mode='valid')
avg_AND_acc_2 = np.convolve(data_fuse_2['acc_AND'], np.ones(window_conv)/window_conv, mode = 'valid')
std_AND_acc_2 = np.convolve(data_std_2['acc_AND'], np.ones(window_conv)/window_conv, mode='valid')

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Accuracy over Time for Task AND for Each Model")
ax.plot(avg_AND_acc_1, color = "blue", label = "our model")
ax.fill_between(trials, avg_AND_acc_1 - 1.96 * std_AND_acc_1, avg_AND_acc_1 + 1.96 * std_AND_acc_1, color='blue', alpha=0.2)
ax.plot(avg_AND_acc_2, color = "grey", label = "random sampling")
ax.fill_between(trials, avg_AND_acc_2 - 1.96 * std_AND_acc_2, avg_AND_acc_2 + 1.96 * std_AND_acc_2, color='grey', alpha=0.2)
ax.set_xlabel("trial")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()
plt.close(fig)


#XOR: compare accuracy over time
avg_XOR_acc_1 = np.convolve(data_fuse_1['acc_XOR'], np.ones(window_conv)/window_conv, mode = 'valid')
std_XOR_acc_1 = np.convolve(data_std_1['acc_XOR'], np.ones(window_conv)/window_conv, mode='valid')
avg_XOR_acc_2 = np.convolve(data_fuse_2['acc_XOR'], np.ones(window_conv)/window_conv, mode = 'valid')
std_XOR_acc_2 = np.convolve(data_std_2['acc_XOR'], np.ones(window_conv)/window_conv, mode='valid')

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_title("Comparison of Accuracy over Time for task XOR for Each Model")
ax.plot(avg_XOR_acc_1, color = 'blue', label = "our model")
ax.fill_between(trials, avg_XOR_acc_1 - 1.96 * std_XOR_acc_1, avg_XOR_acc_1 + 1.96 * std_XOR_acc_1, color='blue', alpha=0.2)
ax.plot(avg_XOR_acc_2, color = "grey", label = "random sampling")
ax.fill_between(trials, avg_XOR_acc_2 - 1.96 * std_XOR_acc_2, avg_XOR_acc_2 + 1.96 * std_XOR_acc_2, color='grey', alpha=0.2)
ax.set_xlabel("trial")
ax.set_ylabel("accuracy")
ax.legend()
plt.show()
plt.close(fig)


###save the model
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''
