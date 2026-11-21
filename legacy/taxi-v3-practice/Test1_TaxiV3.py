import numpy as np
import gym
import random

def main():

    # create Taxi environment
    env = gym.make('Taxi-v3')

    # initialize q-table
    state_size = env.observation_space.n
    action_size = env.action_space.n
    qtable = np.zeros((state_size, action_size))

    # hyperparameters
    learning_rate = 0.9
    discount_rate = 0.8
    epsilon = 1.0
    decay_rate= 0.005

    # training variables
    num_episodes = 5000
    max_steps = 99 # per episode

    # training
    for episode in range(num_episodes):

        # reset the environment
        state = env.reset()
        done = False

        for s in range(max_steps):

            # exploration-exploitation tradeoff
            if random.uniform(0,1) < epsilon:
                # explore
                action = env.action_space.sample()
            else:
                # exploit
                action = np.argmax(qtable[state,:])

            # take action and observe reward
            new_state, reward, done, info = env.step(action)

            # Q-learning algorithm
            qtable[state,action] = qtable[state,action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:])-qtable[state,action])

            # Update to our new state
            state = new_state

            # if done, finish episode
            if done == True:
                break

        # Decrease epsilon
        epsilon = np.exp(-decay_rate*episode)

    print(f"Training completed over {num_episodes} episodes")
    input("Press Enter to watch trained agent...")

    #test result
    test_ep = 20
    step_avg = 0
    score_avg = 0
    failed_factor = 0
    for i in range(test_ep):

        # watch trained agent
        state = env.reset()
        done = False
        rewards = 0

        for s in range(max_steps):

            # print(f"TRAINED AGENT")
            # print("Step {}".format(s+1))

            action = np.argmax(qtable[state,:])
            new_state, reward, done, info = env.step(action)
            rewards += reward
            # env.render()
            # print(f"score: {rewards}")
            state = new_state

            if done == True:
                break
        if s == 98 :
            failed_factor += 1
            print("{0}. FAILED".format(i+1))
            continue

        print("{0}. step: {1}, score: {2}".format(i+1,s,rewards))
        score_avg+= rewards
        step_avg+= s
        
    score_avg /= test_ep
    step_avg /= test_ep
    failed_factor /= test_ep
    print(f"Trained by {num_episodes} episodes")
    print(f"Test under {test_ep} episodes")
    print(f"Average Step: {step_avg}")
    print(f"Average Score: {score_avg}")
    print(f"Failed episode: {failed_factor}")
    env.close()

if __name__ == "__main__":
    main()