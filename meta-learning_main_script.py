#TITLE: META LEARNING MAIN SCRIPT

#IMPORT PACKAGES
import tensorflow as tf
import numpy as np
import pandas as pd
import keras 


#PARAMETERS
n_trials = 900
model_runs = 30
epochs = 1
alpha = 0.3 #learning rate
beta = 1 #reverse temperature
window_size = int(60)
half_window = int(window_size/2)


#X DATA
##initialise the lists for all tasks
train_x = {}
test_x = {}

##set data
x_inputs =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
context_vectors = {
    'AND': np.array([1, 0, 0]),
    'XOR': np.array([0, 1, 0]),
    'RM': np.array([0, 0, 1])
}

##apply data into a shaped array
def apply_x_parameters(context_vectors):
    
    applied_x_parameters = np.concatenate([
        x_inputs,
        np.tile(context_vectors, (4, 1))
    ], axis=1)
    
    return applied_x_parameters

##apply the data to each task's specific parameters
for task, context in context_vectors.items():
    train_x[task] = apply_x_parameters(context)  
    
#Y DATA
##RM (is random)
def RM_y_data(): 
    return np.random.randint(0, 2, size=(4, 1))

##XOR and AND
train_y = {
    'AND': np.array([0, 0, 0, 1]).reshape(4, 1),
    'XOR': np.array([0, 1, 1, 0]).reshape(4, 1), 
    #RM will be randomised every trial through RM_y_data()
    }

#PICK 1 ROW PER TRIAL
def extract_1_row(): 
    train_x_trial = train_x.copy()
    train_y_trial = train_y.copy()
    
    random_rows = {task: np.random.randint(0, 4) for task in train_x.keys()}
    
    for task, data in train_y_trial.items():
        row = random_rows[task]
        train_y_trial[task] = data[row].reshape(1, 1)
        
    for task, data in train_x_trial.items(): 
        row = random_rows[task]
        train_x_trial[task] = data[row].reshape(1, -1)
        test_x[task] = np.copy(train_x_trial[task])
       
    return train_x_trial, train_y_trial

##tracking loss and accuracy
def learning_track():
    test_results_RM = model.evaluate(train_x_trial['RM'], train_y_trial['RM'])
    data.loc[index, 'loss_RM'] = test_results_RM[0]
    data.loc[index, 'acc_RM'] = test_results_RM[1]

    test_results_XOR = model.evaluate(train_x_trial['XOR'], train_y_trial['XOR'])
    data.loc[index, 'loss_XOR'] = test_results_XOR[0]
    data.loc[index, 'acc_XOR'] = test_results_XOR[1]
        
    test_results_AND = model.evaluate(train_x_trial['AND'], train_y_trial['AND'])
    data.loc[index, 'loss_AND'] = test_results_AND[0]
    data.loc[index, 'acc_AND'] = test_results_AND[1]

#initialise parameters
def initialise():
    w = np.array([1/3, 1/3, 1/3]) 
    
    reward_types = ['LP_signed', 'LP_unsigned', 'acc', 'novelty']
    
    weight = {
    'LP_signed': 10,
    'LP_unsigned': 10,
    'acc': 1,
    'novelty': 1
    } #manually set for now
    
    value = {
        a: {types: 0.0 for types in reward_types}
        for a in range(3)
        }
    
    return w, weight, value


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
    loss_history = {0: [], 1: [], 2: []} #keep track of the loss for each action when chosen
    loss_history = {key: [0] * window_size for key in loss_history} #fill the lists with 0's until actual values are given
    w, weight, value = initialise()
    
    
    #MODEL STRUCTURE
    model = keras.models.Sequential([
        keras.layers.core.Dense(8, activation='tanh', input_shape=(5,)),
        keras.layers.core.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05), 
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy()]
        )
    

    #TRIAL LOOP 
    ##action choice
    for i in range(n_trials):
        #record the trial and run
        index += 1
        trial += 1
        data.loc[index, 'loop'] = loop
        data.loc[index, 'trial'] = trial
        print("trial number:", data.loc[index, 'trial'])
         
        #changeable data
        train_y['RM'] = RM_y_data()
        
        #extract 1 row for all tasks to train and test
        train_x_trial, train_y_trial = extract_1_row()
        
        #make a choice (softmax function)
        prob = np.exp(beta*w)/np.sum(np.exp(beta*w))
        data.loc[index, ['prob_RM', 'prob_XOR', 'prob_AND']] = prob
        chosen_task = np.random.choice(3, p = prob)
        
        #record the choice
        data.loc[index, 'chosen_task'] = chosen_task 
        action_counts[chosen_task] += 1
        
    ##set the context
        if chosen_task == 0:
            task_name = 'RM'

        elif chosen_task == 1: 
            task_name = 'XOR'
                  
        elif chosen_task == 2:
            task_name = 'AND'
         
    ##train the model
        history = model.fit(train_x_trial[task_name], train_y_trial[task_name], batch_size = 1, epochs=epochs)
        
    ##record data after training
        learning_track()
              
    ##averaging previous accuracies
        if chosen_task == 0:
            trial_loss = data['loss_RM'].iloc[-1]

        elif chosen_task == 1: 
            trial_loss = data['loss_XOR'].iloc[-1]
                  
        elif chosen_task == 2:
            trial_loss = data['loss_AND'].iloc[-1]

        loss_history[chosen_task].append(trial_loss)        
        if len(loss_history[chosen_task]) > window_size:
            loss_history[chosen_task].pop(0) 
                
    ##reward signals
        #counts
        half1 = np.mean(loss_history[chosen_task][-window_size: -half_window])
        half2 = np.mean(loss_history[chosen_task][-half_window+1:])
        total_count = sum(action_counts.values())
        
        #criteria
        reward = {
            'LP_signed': (half1 - half2),
            'LP_unsigned': abs(half1 - half2),
            'acc': -(trial_loss),
            'novelty': -(action_counts[chosen_task]/total_count)
            }
        
        #record reward
        for r in reward:
            data.loc[index, f"r_{r}"] = reward[r]
        
        #update each value
        for r in reward:
            value[chosen_task][r] += alpha * (reward[r] - value[chosen_task][r])

        ##learning update 
        w[chosen_task] = sum(
            value[chosen_task][r] * weight[r]
            for r in reward
        )
        data.loc[index, 'weight_task'] = w[chosen_task]
        
        
    ##GATHER DATA
    ##test the models
    test_results_AND = model.evaluate(train_x_trial['AND'], train_y_trial['AND'])
    test_predictions_AND = model.predict(train_x_trial['AND'])

    test_results_XOR = model.evaluate(train_x_trial['XOR'], train_y_trial['XOR'])
    test_predictions_XOR = model.predict(train_x_trial['XOR'])

    test_results_RM = model.evaluate(train_x_trial['RM'], train_y_trial['RM'])
    test_predictions_RM = model.predict(train_x_trial['RM'])
    
    ##summary
    print(["Weights:", w])
    print("Action selection frequency:", action_counts)
    print("Most selected action:", max(action_counts, key=action_counts.get))
    print(f"Task name: AND, Loss: {test_results_AND[0]:.4f}, Accuracy: {test_results_AND[1]:.4f}")
    print(f"Task name: XOR, Loss: {test_results_XOR[0]:.4f}, Accuracy: {test_results_XOR[1]:.4f}")
    print(f"Task name: RM, Loss: {test_results_RM[0]:.4f}, Accuracy: {test_results_RM[1]:.4f}")
    
#SAVE DATA
'''data.to_csv('data/dataframe.csv') #hpc'''
'''data.to_csv('D:/ULB/MA2/STAGE2/code/data/dataframe.csv') #local'''

#SAVE MODEL
'''model.save("D:/ULB/MA2/STAGE2/code/internship_curriculum_model")'''
