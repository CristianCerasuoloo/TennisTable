import numpy as np
from client import Client, JOINTS
import sys
import os
import torch
import math
import torch.optim as optim

import buffer
import train as t
from constants import *

class Rewarder:
    def __init__(self):
        self.last_state = None

    def get_reward(self, new_state):
        reward = 0
        reward_string = ""

        playing = new_state[28] # 0: not playing, 1: playing
        waiting = new_state[26]
        waiting_for_ball = new_state[27] 
        bpos = new_state[17:20]
        # print("Ball position:\n ", bpos)
        bvel = new_state[20:23]
        ppos = new_state[11:14]
        pnorm = new_state[14:17]
        field = new_state[32] # 0: our field, 1: opponent field
        my_score = new_state[34]
        opp_score = new_state[35]
        ball_touched = new_state[31]
        old_my_score = self.last_state[34] if self.last_state is not None else 0
        old_opp_score = self.last_state[35] if self.last_state is not None else 0
        old_ball_touched = self.last_state[31] if self.last_state is not None else False

        if playing and not waiting and not waiting_for_ball:

            # # Reward for scoring
            # if my_score > old_my_score:
            #     reward += 100
            # if opp_score > old_opp_score:
            #     reward -= 100

            # Reward for pad position
            # # if the pad is under the ball
            # if ppos[2] < bpos[2]:
            #     reward += 5
            # else:
            #     reward -= 5
            # if the pad is in the right position, i.e. the normal of the pad is the same as the direction of the ball

            # normalize bpos
            # bpos_unit = bpos / np.linalg.norm(bpos)

            # # print("magnitude of bpos = ", np.linalg.norm(bpos_unit))
            # # print("magnitude of pnorm = ", np.linalg.norm(pnorm))

            # # print("|pnorm • bpos_unit| = ", np.abs(np.dot(pnorm, bpos_unit)))
            # if np.dot(pnorm, bpos_unit) < -0.75 :
            #     reward += 10
            #     reward_string += "Reward for pad position: 10\n"
            # else:
            #     reward -= 10

            # the angle between the pad and the ball must be between -90 and 90 degrees, i.è. cross product between the pad normal and the ball velocity must be positive
            # print("cross(pnorm[:2], bvel[:2]) = ", np.cross(pnorm[:2], bvel[:2]))

            
            # Reward for joint position
            # print("|new_state[1] - bpos[0]| = ", np.abs(new_state[1] - bpos[0]))
            # if np.abs(new_state[1] - bpos[0]) < 0.05:
            #     reward += 10
            # else:
            #     reward -= 10


            # Reward for ball position prediction
            # on z-axis g=9.81 m/s^2 acts on the ball => uniform acceleration motion
            # z = z0 + v0z*t - 0.5*g*t^2
            # on x-axis no acceleration => uniform motion
            # x = x0 + v0x*t
            # on y-axis no acceleration => uniform motion
            # y = y0 + v0y*t

            # Reward for touching the ball with the pad
            if not old_ball_touched and ball_touched:
                reward += 50
                reward_string += "Reward for touching the ball: 50\n"

            # Reward for ball moving in opposite direction and it is on the table
            if bvel[1] > 0 and bpos[2] > 0:
                reward += 50
                reward_string += "Reward for ball moving in opposite direction: 50\n"

        
        # # Reward for scoring
        # if my_score > old_my_score:
        #     reward += 500
        #     reward_string += "Reward for scoring: 500\n"

        # if opp_score > old_opp_score:
        #     reward -= 500
        #     reward_string += "Reward for opponent scoring: -500\n"

        self.last_state = new_state

        if reward_string != "":
            print(reward_string)

        return reward

def train(cli):
    ram = buffer.MemoryBuffer(MAX_BUFFER)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device: ", device)
    trainer = t.Trainer(S_DIM, A_DIM, A_MAX, ram, device)

    rewarder = Rewarder()

    j = np.zeros((JOINTS,))
    # Posizione di partenza dei bracci
    j[2] = math.pi
    j[10] = math.pi / 2
    j[5] = math.pi / 2
    j[9] = math.pi / 4

    for _ep in range(MAX_EPISODES):

        while True:
            state = cli.get_state()[:37]

            # When not playing, send default joints
            if not state[28] or state[26] or state[27]:
                cli.send_joints(j)
                continue

            action = trainer.get_exploration_action(state[11:26])

            cli.send_joints(action)

            new_state = cli.get_state()[:37]
            reward = rewarder.get_reward(new_state) # TODO
            done = is_episode_done(new_state)

                
            # push this exp in ram
            ram.add(state[11:26], action, reward, new_state[11:26])

            trainer.optimize()
            if done:
                break

        # save model
        if _ep % 100 == 0:
            trainer.save_models(_ep)

def is_episode_done(state):
    # one of the players has scored 11 points
    if state[34] % 11 == 0 or state[35] % 11 == 0:
        return True

    # simulation time is over 5 minutes
    # if state[36] > 300:
    #     return True
    
    return False

def main():
    name = 'Full RL Client'
    if len(sys.argv) > 1:
        name = sys.argv[1]

    host = 'localhost'
    if len(sys.argv) > 2:
        host = sys.argv[2]

    cli = Client(name, host)
    train(cli)


if __name__ == '__main__':
    main()