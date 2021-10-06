import gym
import gym_minigrid
import numpy as np
import random
import matplotlib.pyplot as plt

#constants:
state_space = 5
action_space = 3 
directions = 4
actions = [0, 1, 2]

#making the env
env = gym.make('MiniGrid-Empty-6x6-v0')

#global variales:
epsilon = 1
total_episodes = 500
alpha = 0.5
gamma = 0.96
decay = 1.05

#defining q matrix- 4D with x, y, dir, action
q = np.zeros((state_space, state_space, directions, action_space))

def epsilon_greedy(x, y, dir1):
    action = 0
    
    #explore:
    if np.random.uniform(0,1) < epsilon:
        action = random.choice(actions)
        
    #greedy policy:
    else:
        action = np.argmax(q[x, y, dir1, :])
        
    return action

def update(x1, y1, dir1, x2, y2, dir2, reward, action, action2):
    #bootstrapped value:
    predict = q[x1, y1, dir1, action]
    
    target = reward + gamma * q[x2, y2, dir2, action2]
    #update:
    q[x1, y1, dir1, action] = q[x1, y1, dir1, action] + alpha * (target - predict)
    
#to store values for graph:    
graph_ep = []
graph_rew = []
graph_steps = []

#start
for episode in range(total_episodes):
    ep_reward = 0
    print("******episode ",episode, " ********")
    steps = 0
    env.reset()
    x1 ,y1 = env.agent_pos
    dir1 = env.agent_dir
    action1 = epsilon_greedy(x1 ,y1, dir1)
    
    done = False
    
    while(not done):
        #env.render()
        
        state, rew, done, _ = env.step(action1)
        x2, y2 = env.agent_pos
        dir2 = env.agent_dir
        
        #Choosing the next action
        action2 = epsilon_greedy(x2, y2, dir2)
        
        #Learning the Q-value
        update(x1, y1, dir1, x2, y2, dir2, rew, action1, action2)
        
        x1 = x2
        y1 = y2
        dir1= dir2
        action1 = action2
        
        steps += 1
        ep_reward += rew
        
    #decaying epsilon after each episode
    epsilon /= decay
    
    print("Episode reward:", rew)
      
    graph_ep.append(episode)
    graph_rew.append(ep_reward)
    graph_steps.append(steps)
      
print ("Performance : ", sum(graph_rew)/total_episodes)

figure, axis = plt.subplots(1, 2)
plt.figure(figsize=(4, 11), dpi=80)

# For episode vs reward
axis[0].plot(graph_ep, graph_rew)
axis[0].set_title("Episode vs Reward")
 
# For episode vs steps
axis[1].plot(graph_ep , graph_steps)
axis[1].set_title("Episode vs Steps")

# Combine all the operations and display
plt.show()