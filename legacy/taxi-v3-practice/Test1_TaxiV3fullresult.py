import numpy as np
import gym
import random
import pandas as pd

def main ():

    # training variables
    ep_parameter = [200,800,1000,2500,3000]
    max_steps = 99 # per episode  
    test_ep = 5

    # create Taxi environment
    env = gym.make('Taxi-v3')

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005
    
    # collect test result
    test_name_str = ["test{0}".format(i) for i in range(1,test_ep+1)]
    test_result_df = pd.DataFrame(col=test_name_str)
    test_result_df["num of ep"] = ep_parameter
    print(test_result_df.values)
    print("success df value")

    # #training & testing
    # for num_episodes in ep_parameter:
        
    #     # initialize q-table
    #     state_size = env.observation_space.n
    #     action_size = env.action_space.n
    #     qtable = np.zeros((state_size, action_size))
        
    #     #training
    #     for episode in range(num_episodes):
    #         # reset the environment
    #         state = env.reset()
    #         done = False

    #         for s in range(max_steps):

    #             # exploration-exploitation tradeoff
    #             if random.uniform(0,1) < epsilon:
    #                 # explore
    #                 action = env.action_space.sample()
    #             else:
    #                 # exploit
    #                 action = np.argmax(qtable[state,:])

    #             # take action and observe reward
    #             new_state, reward, done, info = env.step(action)

    #             # Q-learning algorithm
    #             qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

    #             # Update to our new state
    #             state = new_state

    #             # if done, finish episode
    #             if done == True:
    #                 break

    #         # Decrease epsilon
    #         epsilon = np.exp(-decay_rate*episode)







    # test_result = np.zeros(len(ep_parameter),num_)