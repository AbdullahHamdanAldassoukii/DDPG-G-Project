import numpy as np
from ddpg import *
import gc
import csv
from Humanoid import Humanoid

gc.enable()

ENV_ID = 'bioloid-v0'
EPISODES = 50000
# EPISODES = 100
TEST = 10
STATE_DIM, ACTION_DIM = 32, 10
MAX_STEPS_PER_EPS = 120

# this file contains the reward over all episodes
reward_file = "reward_file.csv"
trajectory_file = "trajectory_file.csv"

continue_eps=2500
save_eps=continue_eps+100

# the main functionality for the DDPG Algorithm
def main():
    best_reward = 16.0
    # writing rewards in the csv file

    file = open(reward_file, 'a')
    writer = csv.writer(file)

    env = Humanoid()
    env_dim = [STATE_DIM, ACTION_DIM]
    agent = DDPG(env_dim)


    agent.actor_network.load_network(continue_eps)
    agent.critic_network.load_network(continue_eps)


    # main loop
    for episode in range(continue_eps+1,EPISODES):


        state= env.reset()
        for steps in range(MAX_STEPS_PER_EPS):
            action = agent.noise_action(state)

            next_state, reward, done, _ = env.step(action)
            agent.perceive(state, action, reward, next_state, done)
            # env.unpause()
            state = next_state
            if steps >=MAX_STEPS_PER_EPS-1:
                done=True
            if done:
                print("Episode "+str(episode)+" : steps count = "+str(steps) +" , reward ="+str(reward) )
                break

        # Testing:
        if episode % 50 == 0 and episode != 0:
            traj_file = open(trajectory_file, 'wt')
            traj_writer = csv.writer(traj_file, delimiter='\t')
            traj_writer.writerow(
                ['gx', 'gy', 'gz', 'vx', 'vy', 'vz', 'wx', 'wy', 'wz', 'q 9', 'q 10', 'q 11', 'q 12', 'q 13', 'q 14',
                 'q 15', 'q 16', 'q 17', 'q 18', 'qd 9', 'qd 10', 'qd 11', 'qd 12', 'qd 13', 'qd 14', 'qd 15', 'qd 16',
                 'qd 17', 'qd 18', 'tc1', 'tc2', 'duration'])

            print("testing...")
            total_reward = 0
            count_of_1 = 0
            agent.actor_network.save_network(episode)
            agent.critic_network.save_network(episode)

            for i in range(TEST):

                state = env.reset()
                for steps in range(MAX_STEPS_PER_EPS):
                    action = agent.action(state)  # direct action for test
                    next_state, reward, done, _ = env.step(action)
                    traj_writer.writerow(state)

                    traj_file.flush()

                    # todo : find what this for
                    # if reward == 1:
                    #     count_of_1 += 1
                    total_reward += reward
                    if steps >= MAX_STEPS_PER_EPS-1:
                        done = True
                    if done:
                        # print("Episode TEST finished !")
                        break

            ave_reward = total_reward / TEST

            # env.latest_reward = ave_reward
            if ave_reward > best_reward:
                best_reward = ave_reward

            # env.avg_reward = ave_reward
            writer.writerow([ave_reward])
            file.flush()

            print("episode: ", episode, "Evaluation Average Reward: ", ave_reward)
            print("best_reward: ", best_reward)

    pass

def read_csv(file):
    with open(file, 'r') as csvFile:
        reader = csv.reader(csvFile,delimiter='\t')
        lineNum = 0
        ret=[]
        for row in reader:
            if lineNum % 2 == 0:
                ret.append(row)
            lineNum += 1

    csvFile.close()
    return ret

if __name__ == "__main__":
    print("starting")
    main()
    print("khalas")
