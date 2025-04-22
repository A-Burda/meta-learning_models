#TITLE: MODEL DATA ANALYSIS SCRIPT
#IMPORT PACKAGES
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt 
import keras 

#PARAMETERS
window_conv = 10
comparison = True
 
#LOAD DATA
data_main = pd.read_csv('data/dataframe.csv')

def fuse_data(): 
    ###get mean and std
    relevant_data = ['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']
    data_fuse = data.groupby('trial')[relevant_data].mean()
    data_fuse['mean_acc'] = data_fuse.loc[:, ['acc_AND', 'acc_XOR', 'acc_RM']].mean(axis=1)
    data_fuse['mean_loss'] = data_fuse.loc[:, ['loss_AND', 'loss_XOR', 'loss_RM']].mean(axis=1)
    
    ###get percentiles
    percentiles_dict = {measurement: {'p5': [], 'p95': []} for measurement in relevant_data}
    for trial in data_fuse.index:
        
        #extract the data for the trial
        trial_data = data_main[data_main['trial'] == trial]
        
        #calculate the percentiles and add them to the list
        for measurement in relevant_data:
            p5 = np.percentile(trial_data[measurement], 5)
            p95 = np.percentile(trial_data[measurement], 95)
            percentiles_dict[measurement]['p5'].append(p5)
            percentiles_dict[measurement]['p95'].append(p95)
            
    #convolve
    for measurement in relevant_data: 
        p5_array = np.array(percentiles_dict[measurement]['p5'])
        p95_array = np.array(percentiles_dict[measurement]['p95'])      
        percentiles_dict[measurement]['p5'] = np.convolve(p5_array, np.ones(window_conv)/window_conv, mode='valid')
        percentiles_dict[measurement]['p95'] = np.convolve(p95_array, np.ones(window_conv)/window_conv, mode='valid')
        
    ###action probability over time
    task = ['prob_RM', 'prob_XOR', 'prob_AND']
    con_fuse = {}
    
    for task in task:
        con_fuse[task] = np.convolve(data_fuse[task], np.ones(window_conv)/window_conv, mode='valid')
        
    task_acc = ['acc_RM', 'acc_XOR', 'acc_AND']
    acc_avg = {}
    
    for task_acc in task_acc: 
        acc_avg[task_acc] = np.convolve(data_fuse[task_acc], np.ones(window_conv)/window_conv, mode = 'valid')
        
    task_loss = ['loss_RM', 'loss_XOR', 'loss_AND']
    loss_avg = {}
    
    for task_loss in task_loss: 
        loss_avg[task_loss] = np.convolve(data_fuse[task_loss], np.ones(window_conv)/window_conv, mode = 'valid')
        
    trials = np.arange(len(con_fuse['prob_RM']))
    
    return trials, con_fuse, acc_avg, loss_avg, percentiles_dict
    
def plot_data():
    #choice probabilities over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Choice Probabilities over Time")
    ax.plot(con_fuse['prob_RM'], color='red', label = "RM")
    ax.fill_between(trials, percentiles_dict['prob_RM']['p5'], percentiles_dict['prob_RM']['p95'], color='red', alpha=0.2)
    ax.plot(con_fuse['prob_XOR'], color='green', label = "XOR")
    ax.fill_between(trials, percentiles_dict['prob_XOR']['p5'], percentiles_dict['prob_XOR']['p95'], color='green', alpha=0.2)
    ax.plot(con_fuse['prob_AND'], color='blue', label = "AND")
    ax.fill_between(trials, percentiles_dict['prob_AND']['p5'], percentiles_dict['prob_AND']['p95'], color='blue', alpha=0.2)
    ax.set_xlabel("trial")
    ax.set_ylabel("choice probability")
    ax.legend()
    plt.show()
    plt.close(fig)
    
    #accuracy over time 
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Comparison of Accuracy over Time for Each Task")
    ax.plot(acc_avg['acc_AND'], color = "blue", label = "AND")
    ax.fill_between(trials, percentiles_dict['acc_AND']['p5'], percentiles_dict['acc_AND']['p95'], color='blue', alpha=0.2)
    ax.plot(acc_avg['acc_XOR'], color = "green", label = "XOR")
    ax.fill_between(trials, percentiles_dict['acc_XOR']['p5'], percentiles_dict['acc_XOR']['p95'], color='green', alpha=0.2)
    ax.plot(acc_avg['acc_RM'], color = "red", label = "RM")
    ax.fill_between(trials, percentiles_dict['acc_RM']['p5'], percentiles_dict['acc_RM']['p95'], color='red', alpha=0.1)
    ax.set_xlabel("trial")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.show()
    plt.close(fig)
    
    #loss over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Comparison of Loss Functions for Each Task")
    ax.plot(loss_avg['loss_AND'], color = "blue", label = "AND")
    ax.fill_between(trials, percentiles_dict['loss_AND']['p5'], percentiles_dict['loss_AND']['p95'], color='blue', alpha=0.2)
    ax.plot(loss_avg['loss_XOR'], color = "green", label = "XOR")
    ax.fill_between(trials, percentiles_dict['loss_XOR']['p5'], percentiles_dict['loss_XOR']['p95'], color='green', alpha=0.2)
    ax.plot(loss_avg['loss_RM'], color = "red", label = "RM")
    ax.fill_between(trials, percentiles_dict['loss_RM']['p5'], percentiles_dict['loss_RM']['p95'], color='red', alpha=0.1)
    ax.set_xlabel("trial")
    ax.set_ylabel("loss")
    ax.legend()
    plt.show()
    plt.close(fig)

def model_comparison():    
    #AVG: compare accuracy over time
    '''
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Comparison of Accuracy over Time Across Tasks for Each Model")
    ax.plot(acc_avg, color="blue", label='our model')
    ax.fill_between(trials, acc_avg - 1.96 * acc_std, acc_avg + 1.96 * acc_std, color='blue', alpha=0.1)
    ax.plot(acc_avg_iid, color="grey", label='random sampling')
    ax.fill_between(trials, acc_avg_iid - 1.96 * acc_std_iid, acc_avg_iid + 1.96 * acc_std_iid, color='grey', alpha=0.1)
    ax.set_xlabel("trial")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.show()
    plt.close(fig)'''

    #AND: compare accuracy over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Comparison of Accuracy over Time for Task AND for Each Model")
    ax.plot(acc_avg['acc_AND'], color = "blue", label = "our model")
    ax.fill_between(trials, percentiles_dict['acc_AND']['p5'], percentiles_dict['acc_AND']['p95'], color='blue', alpha=0.2)
    ax.plot(acc_avg_iid['acc_AND'], color = "grey", label = "random sampling")
    ax.fill_between(trials, percentiles_dict_iid['acc_AND']['p5'], percentiles_dict_iid['acc_AND']['p95'], color='grey', alpha=0.2)
    ax.set_xlabel("trial")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.show()
    plt.close(fig)


    #XOR: compare accuracy over time
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_title("Comparison of Accuracy over Time for task XOR for Each Model")
    ax.plot(acc_avg['acc_XOR'], color = 'blue', label = "our model")
    ax.fill_between(trials, percentiles_dict['acc_XOR']['p5'], percentiles_dict['acc_XOR']['p95'], color='blue', alpha=0.2)
    ax.plot(acc_avg_iid['acc_XOR'], color = "grey", label = "random sampling")
    ax.fill_between(trials, percentiles_dict_iid['acc_XOR']['p5'], percentiles_dict_iid['acc_XOR']['p95'], color='grey', alpha=0.2)
    ax.set_xlabel("trial")
    ax.set_ylabel("accuracy")
    ax.legend()
    plt.show()
    plt.close(fig)

#RUN DATA 
data = data_main 
trials, con_fuse, acc_avg, loss_avg, percentiles_dict = fuse_data()
plot_data()

if comparison: 
    data_iid = pd.read_csv('data/dataframe_iid.csv')
    data = data_iid
    trials, con_fuse_iid, acc_avg_iid, loss_avg_iid, percentiles_dict_iid = fuse_data()
    model_comparison()
