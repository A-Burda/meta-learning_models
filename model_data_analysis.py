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
comparison = False

#SIMPLE MODEL ANALYSIS 
##load data
data_main = pd.read_csv('E:/ULB/MA2/STAGE2/code/data/dataframe.csv')

###get mean and std
data_fuse = data_main.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].mean()
data_std = data_main.groupby('trial')[['prob_RM', 'prob_XOR', 'prob_AND', 'acc_RM', 'acc_XOR', 'acc_AND', 'loss_RM', 'loss_XOR', 'loss_AND']].std()

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


#IIC MODEL ANALYSIS (WIP)
def random_loop():
    data_main = pd.read_csv('E:/ULB/MA2/STAGE2/code/data/dataframeII.csv')

if comparison == True :
    random_loop()

#MODEL COMPARISON (WIP)
