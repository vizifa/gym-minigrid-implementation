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
total_episodes = 300
alpha = 0.35
gamma = 0.8
decay = 1.04

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

def update(x1, y1, dir1, x2, y2, dir2, reward, action):
    #bootstrapped value:
    predict = q[x1, y1, dir1, action]
    
    target = reward + gamma * np.argmax(q[x2, y2, dir2, :])
    #update:
    q[x1, y1, dir1, action] = q[x1, y1, dir1, action] + alpha * (target - predict)
    
#to store values for the graph   
graph_ep = []
graph_rew = []
graph_steps = []

#start of episodes
for episode in range(total_episodes):
    ep_reward = 0
    print("******Episode ", episode, " ********")
    
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
        
        #Learning the Q-value
        update(x1, y1, dir1, x2, y2, dir2, rew, action1)
        
        x1 = x2
        y1 = y2
        dir1= dir2
        action1 = epsilon_greedy(x1 ,y1, dir1)
        
        steps += 1
        ep_reward += rew
    
    #decaying epsilon after each episode
    epsilon /= decay
    
    print("Episode reward:", ep_reward)
      
    graph_ep.append(episode)
    graph_rew.append(ep_reward)
    graph_steps.append(steps)
      
print ("Performance : ", sum(graph_rew)/total_episodes)
env.close()

#Plotting graphs
figure, axis = plt.subplots(1, 2)
  
#For Episodes vs Reward
axis[0].plot(graph_ep, graph_rew)
axis[0].set_title("Episodes vs Reward")
    
#For Episodes vs Steps
axis[1].plot(graph_ep , graph_steps)
axis[1].set_title("Episodes vs Steps")
plt.show()