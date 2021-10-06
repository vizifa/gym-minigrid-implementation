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
alpha = 0.3
gamma = 0.9
decay = 1.05
lamda = 0.9

#defining q matrix- 4D with x, y, dir, action
q = np.zeros((state_space, state_space, directions, action_space))
e = np.zeros((state_space, state_space, directions, action_space))

def epsilon_greedy(x, y, dir1):
    action = 0
    
    #explore:
    if np.random.uniform(0,1) < epsilon:
        action = random.choice(actions)
        
    #greedy policy:
    else:
        action = np.argmax(q[x, y, dir1, :])
        
    return action

def updateQ(delta):

    for x in range(state_space):
        for y in range(state_space):
            for d in range(directions):
                for a in actions:
                    q[x, y, d, a] += alpha * delta * e[x, y, d, a]
                    e[x, y, d, a] = gamma * lamda * e[x, y, d, a]

    
    
graph_ep = []
graph_rew = []
graph_steps = []

for episode in range(total_episodes):
    
    #defining eligibility traces
    e = np.zeros((state_space, state_space, directions, action_space))
    
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
        
        predict = q[x1, y1, dir1, action1]
        target = q[x2, y2, dir2, action2]
        delta = rew + gamma*target - predict
        
        #updating eligibility traces:
        e[x1, y1, dir1, action1] = 1
        
        #Learning the Q-value
        updateQ(delta)
        
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
env.render(close=True)

print("Graphs:")

#Plotting graphs
figure, axis = plt.subplots(1, 2)
  
#For Episodes vs Reward
axis[0].plot(graph_ep, graph_rew)
axis[0].set_title("Episodes vs Reward")
    
#For Episodes vs Steps
axis[1].plot(graph_ep , graph_steps)
axis[1].set_title("Episodes vs Steps")
plt.show()