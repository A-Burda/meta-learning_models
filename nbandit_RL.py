#INITIALISATION
import numpy as np
import matplotlib.pyplot as plt

#PARAMETERS
n_trials = 500
alpha = 0.1 #learning rate
beta = 0.7 #reverse temperature
p = [0.2, 0.4, 0.6, 0.8] #payoff probabilities
w = np.random.random(len(p)) #weights

#DATA START
trial = []
reward_list = [] 
choice_list = []
action_list = list(range(len(p)))
action_counts = {action: 0 for action in action_list}

#ACTION CHOICE
for i in range(n_trials): 
    trial.append(i + 1) #counting the trials
    prob = np.exp(beta*w)/np.sum(np.exp(beta*w)) #softmax function
    choice = np.random.choice(len(p), p = prob)
    choice_list.append(choice) 
    action_counts[choice] += 1
#LEARNING UPDATE
    reward = np.random.choice([0, 1], p =  [1 - p[choice], p[choice]]) #receive reward
    reward_list.append(reward)
    w[choice] += alpha*(reward - w[choice]) #Rescorla-Wagner learning rule
    
#REPORT DATA
##summary
print(["Weights:", w])
print("Action selection frequency:", action_counts)
print("Most selected action:", max(action_counts, key=action_counts.get))
print("mean reward: {:.2f}".format(np.mean(reward_list)))

##plots
window_conv = 20
avg_choices = np.convolve(choice_list, np.ones(window_conv)/window_conv, mode = 'valid')
avg_reward = np.convolve(reward_list, np.ones(window_conv)/window_conv, mode = 'valid')
x_values = np.arange(len(avg_choices))
slope_choices, intercept_choices = np.polyfit(x_values, avg_choices, 1)
slope_rewards, intercept_rewards = np.polyfit(x_values, avg_reward, 1)
trend_choices = slope_choices * x_values + intercept_choices
trend_rewards = slope_rewards * x_values + intercept_rewards

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(avg_reward *3, color='blue', alpha = 0.3)
ax.plot(avg_choices, color='black', alpha = 0.5)
ax.plot(trend_choices, color='black', linestyle="dashed")
ax.plot(trend_rewards *3, color='blue', linestyle="dashed")
ax.set_xlabel("n_trial")
ax.set_ylabel("choice")
ax.set_yticks(range(len(p)))
plt.show()