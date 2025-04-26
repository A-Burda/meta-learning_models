#TITLE: META LEARNING MAIN SCRIPT
#IMPORT PACKAGES
import tensorflow as tf
import numpy as np
import pandas as pd
import keras 


###############################################################################
#PARAMETERS
reward_types = ['LP_signed', 'LP_unsigned', 'acc', 'novelty']

'''weight_task = {}
for types in reward_types:
    weight_task[types] = 1''' #for later
    
weight_task = {
    'LP_signed': 10,
    'LP_unsigned': 10,
    'acc': 1,
    'novelty': 1
    } #manually set for now

##general parameters
iid_sampling = False

parameters = {
    'iid_sampling': False,
    'n_trials': 900,
    'model_runs': 30,
    'epochs': 1,
    'alpha': 0.3, #learning rate
    'beta': 1, #reverse temperature
    'window_size': int(40),
    'half_window': int(20)
    }

###############################################################################
#CREATE DATA SETS
##x data set
train_x = {}
test_x = {}

###base x data
x_inputs =  np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

###context data
context_vectors = {
    'AND': np.array([1, 0, 0]),
    'XOR': np.array([0, 1, 0]),
    'RM': np.array([0, 0, 1])
}

###shaping data
def apply_x_parameters(context_vectors):
    
    applied_x_parameters = np.concatenate([
        x_inputs,
        np.tile(context_vectors, (4, 1))
    ], axis=1)
    
    return applied_x_parameters

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

###############################################################################
#DATA MANAGMENT
##initialisations
def initialise_all(parameters):
    data = pd.DataFrame()
    loop = 0
    index = 0
    if iid_sampling: 
        parameters['beta'] = 0 
        
    return data, loop, index, parameters

def initialise_run(loop, reward_types):
    trial = 0
    loop += 1
    action_list = list(range(3))
    action_counts = {action: 0 for action in action_list}
    #keep track of the loss for each action when chosen
    loss_history = {0: [], 1: [], 2: []} 
    #fill the lists with 0's until actual values are given
    loss_history = {key: [0] * parameters['window_size'] for key in loss_history} 
    w = np.array([1/3, 1/3, 1/3]) 
    
    '''weight_task = weight_task''' #weights will be reset here when using policy gradient (import and export weight_task)
    
    value = {
        a: {types: 0.0 for types in reward_types}
        for a in range(3)
        }
    
    return trial, loop, action_counts, loss_history, w, value

def initialise_trial(data, index, trial, loop):
    index += 1
    trial += 1
    data.loc[index, 'loop'] = loop
    data.loc[index, 'trial'] = trial
    
    return index, trial
    
##tracking data
def learning_track(train_x_trial, train_y_trial, index, trial, data, model):
    test_results = {}
    
    for task in train_x_trial:
        test_results[task] = model.evaluate(train_x_trial[task], train_y_trial[task])
        data.loc[index, f'loss_{task}'] = test_results[task][0]
        data.loc[index, f'acc_{task}'] = test_results[task][1]
        
    return test_results

##model predictions
def predict_model(train_x_trial, model):
    test_predictions = {}

    for task in train_x_trial:
        test_predictions = model.predict(train_x_trial[task])
            
    return test_predictions

##save data
def save_data(data): 
    if iid_sampling: 
        data.to_csv('data/dataframe_iid.csv')
    else: 
        data.to_csv('data/dataframe.csv')
        

###############################################################################
#NEURAL MODEL COMPONENTS
def neural_model():
    model = keras.models.Sequential([
        keras.layers.core.Dense(8, activation='tanh', input_shape=(5,)),
        keras.layers.core.Dense(1, activation='sigmoid')
        ])

    model.compile(
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.05), 
        loss = tf.keras.losses.BinaryCrossentropy(),
        metrics = [tf.keras.metrics.BinaryAccuracy()]
        )
    
    return model

###############################################################################
#REINFORCEMENT LEARNING MODEL COMPONENTS
#make a choice (softmax function)
def get_choice(w, index, action_counts, data): 
    prob = np.exp(parameters['beta']*w)/np.sum(np.exp(parameters['beta']*w))
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
    
    return chosen_task, action_counts, task_name
    
def get_reward(data, index, chosen_task, task_name, action_counts, loss_history):
    #averaging previous accuracies
    trial_loss = data[f'loss_{task_name}'].iloc[-1]

    loss_history[chosen_task].append(trial_loss)        
    if len(loss_history[chosen_task]) > parameters['window_size']:
        loss_history[chosen_task].pop(0) 
            
##reward signals
    #counts
    half1 = np.mean(loss_history[chosen_task][-parameters['window_size']: -parameters['half_window']])
    half2 = np.mean(loss_history[chosen_task][-parameters['half_window']+1:])
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
     
    return reward

def get_value(chosen_task, value, reward):
    for r in reward:
        value[chosen_task][r] += parameters['alpha'] * (reward[r] - value[chosen_task][r])
    
    return value 
    
def learning_update(data, index, chosen_task, w, weight_task, value, reward):
    w[chosen_task] = sum(
        value[chosen_task][r] * weight_task[r]
        for r in reward
    )
    data.loc[index, 'weight_task'] = w[chosen_task]
    
    return w

###############################################################################
#MAIN LOOP
data, loop, index, parameters = initialise_all(parameters)

#MODEL LOOP
for run in range(parameters['model_runs']):
    trial, loop, action_counts, loss_history, w, value = initialise_run(loop, reward_types)
    
    #compile neural model
    model = neural_model()
    
    #TRIAL LOOP 
    ##action choice
    for i in range(parameters['n_trials']):
        index, trial = initialise_trial(data, index, trial, loop)
         
        #changeable data
        train_y['RM'] = RM_y_data()
        
        #extract 1 row for all tasks to train and test
        train_x_trial, train_y_trial = extract_1_row()
        
        #make a choice
        chosen_task, action_counts, task_name = get_choice(w, index, action_counts, data)
         
        #train the model
        history = model.fit(train_x_trial[task_name], train_y_trial[task_name], batch_size = 1, epochs=parameters['epochs'])
        
        #record data after training
        learning_track(train_x_trial, train_y_trial, index, trial, data, model)
              
        #calculating reward
        reward = get_reward(data, index, chosen_task, task_name, action_counts, loss_history)
        
        value = get_value(chosen_task, value, reward)

        ##learning update 
        w = learning_update(data, index, chosen_task, w, weight_task, value, reward)
            
    ##GATHER DATA
    test_results = learning_track(train_x_trial, train_y_trial, index, trial, data, model)    
    test_predictions = predict_model(train_x_trial, model)    
                
    #SAVE DATA
    save_data(data)
