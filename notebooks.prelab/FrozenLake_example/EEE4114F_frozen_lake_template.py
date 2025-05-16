import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

gym.register(
    id="FrozenLake-enhanced", # give it a unique id
    entry_point="frozen_lake_enhanced:FrozenLakeEnv", # frozen_lake_enhanced = name of file 'frozen_lake_enhanced.py'
    kwargs={"map_name": "4x4"},
    max_episode_steps=100,
    reward_threshold=0.85,  # optimum = 0.91
)

def run(episodes, is_training=True, render=False):
    env = gym.make('FrozenLake-enhanced', desc=None, map_name="4x4", is_slippery=False, render_mode='human' if render else None)

    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n)) # init a 16 x 4 array
    else:
        f = open('frozen_lake4x4.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    rng = np.random.default_rng()   # random number generator
    alpha = 0.1 #learning rate
    gama = 0.9 # discount rate
    epsilon = 0.9 #

    rewards_per_episode = np.zeros(episodes) # Otherwise known as the return

    for i in range(episodes):
        state = env.reset()[0]  # states: 0 to 15, 0=top left corner,15=bottom right corner
        terminated = False      # True when fall in hole or reached goal
        truncated = False       # True when actions > 100

        while(not terminated and not truncated):
            if(is_training and rng.random() < epsilon):
                action = env.action_space.sample() # action: 0=left, 1=down, 2=right, 2=up
            else:
                action = np.argmax(q[state,:])


            new_state, reward, terminated, truncated,_ = env.step(action)

            if(is_training):
                q[state, action] = q[state, action]+alpha*(
                    reward + gama * np.max(q[new_state, :]) - q[state, action]
                )

                state = new_state

            # pass the q table and episode count to the environment for rendering
            if(env.render_mode=='human'):
                env.set_q(q)
                env.set_episode(i)

                if reward == 1:
                    rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.savefig('frozen_lake4x4.png')

    if is_training:
        f = open("frozen_lake4x4.pkl","wb")
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(1000, is_training=False, render=True)