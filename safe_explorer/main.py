import time
from datetime import datetime
from functional import seq
import numpy as np
import torch
import os
from safe_explorer.core.config import Config
from safe_explorer.core.tensorboard import TensorBoard
from safe_explorer.env.ballnd import BallND
from safe_explorer.env.spaceship import Spaceship
from safe_explorer.safety_layer.safety_layer import SafetyLayer
from safe_explorer.guide.guide import Guide
from safe_explorer.ddpg.ddpg import DDPGAgent
from safe_explorer.ddpg.utils import OUNoise
import csv
import time

class Trainer:
    def __init__(self):
        config = Config.get().main.trainer
        # set seeds
        torch.manual_seed(config.seed)
        np.random.seed(int(config.seed))

        self.use_safety_layer = config.use_safety_layer
        
        # create environment
        if config.task == 'ballnd':
            self.env = BallND()
        else:
            self.env = Spaceship()

        self.train_safety_counter = 0
        self.eval_safety_counter = 0
        self.train_guidance_counter = 0
        self.eval_guidance_counter = 0      
        
        # ts stores the time in seconds
        self.ts = time.time()
        self.ts = int(self.ts)
        if not os.path.exists('runs/' + str(self.ts)):
            os.makedirs('runs/' + str(self.ts))
        header = ['train_step', 'train guidance replacements', 'train safety replacements']
        with open('runs/' + str(self.ts) + '/train_data.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

        header = ['eval_step', 'episode length', 'episode reward', 'cumulative constraint violations', 'eval guidance replacements', 'eval safety replacements']
        with open('runs/' + str(self.ts) + '/eval_data.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

        header = ['time']
        with open('runs/' + str(self.ts) + '/time.csv', 'w', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(header)
            f.close()

    def train(self):
        print("============================================================")
        print("Initialized SafeExplorer with config:")
        print("------------------------------------------------------------")
        Config.get().pprint()
        print("============================================================")

        # init Safety Layer
        safety_layer = None
        if self.use_safety_layer:
            safety_layer = SafetyLayer(self.env)
            safety_layer.train()

        guide = None
        guide = Guide(self.env)
        # obtain state and action dimensions
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]

        # get config
        config = Config.get().ddpg.trainer
        # get relevant config values
        epochs = config.epochs
        training_episodes = config.training_episodes_per_epoch
        evaluation_episodes = config.evaluation_episodes_per_epoch
        # max_episode_length = config.max_episode_length
        batch_size = config.batch_size

        # create agent
        agent = DDPGAgent(state_dim, action_dim, self.ts)
        # create exploration noise
        noise = OUNoise(self.env.action_space)
        # metrics for tensorboard
        cum_constr_viol = 0  # cumulative constraint violations
        eval_step = 0
        train_step = 0
        # create Tensorboard writer
        writerTB = TensorBoard.get_writer()

        start_time = time.time()
        print("==========================================================")
        print("Initializing DDPG training...")
        print("----------------------------------------------------------")
        print(f"Start time: {datetime.fromtimestamp(start_time)}")
        print("==========================================================")

        for epoch in range(epochs):
            # training phase
            agent.set_train_mode()
            for _ in range(training_episodes):
                noise.reset()
                state = self.env.reset()
                done = False
                while not done:
                    # get original policy action
                    action = agent.get_action(state)
                    # add OU-noise
                    action = noise.get_action(action)

                    old_action = action
                    #print('action before guidance: ', action)
                    action = guide.get_guided_action(action, _)
                    #print('action after guidance: ', action)
                    if old_action != action:
                        self.train_guidance_counter += 1
                    
                    old_action = action
                    # print('action before safety: ', action)

                    # get safe action
                    if safety_layer:
                        constraints = self.env.get_constraint_values()
                        action = safety_layer.get_safe_action(
                            state, action, constraints)
                    #print('action after safety: ', action)
                    if old_action != action:
                        self.train_safety_counter += 1 

                    # apply action
                    next_state, reward, done, _ = self.env.step(action)
                    # push to memory
                    agent.memory.push(state, action, reward, next_state, done)
                    # update agent
                    if len(agent.memory) > batch_size:
                        agent.update(batch_size)
                    state = next_state
                writerTB.add_scalar("metrics/train guidance replacements", self.train_guidance_counter, train_step)
                writerTB.add_scalar("metrics/train safety replacements", self.train_safety_counter, train_step)

                data = [train_step, self.train_guidance_counter,  self.train_safety_counter]
                with open('runs/' + str(self.ts) + '/train_data.csv', 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                    f.close()

                train_step += 1
            print(f"Finished epoch {epoch}. Running evaluation ...")

            # evaluation phase
            agent.set_eval_mode()
            episode_rewards, episode_lengths, episode_actions = [], [], []
            for _ in range(evaluation_episodes):
                state = self.env.reset()
                episode_action, episode_reward, episode_step = 0, 0, 0
                done = False
                # render environment
                # self.env.render()
                while not done:
                    # get original policy action
                    action = agent.get_action(state)
                    old_action = action
                    #print('action before guidance: ', action)

                    # get safe action
                    # action = guide.get_guided_action(action, _)
                    #print('action after guidance: ', action)

                    if old_action != action:
                        self.eval_guidance_counter += 1
                    
                    old_action = action
                    #print('action before safety: ', action)

                    if safety_layer:
                        constraints = self.env.get_constraint_values()
                        action = safety_layer.get_safe_action(
                            state, action, constraints)
                    #print('action after safety: ', action)

                    if old_action != action:
                        self.eval_safety_counter += 1 

                    episode_action += np.absolute(action)
                    # apply action
                    state, reward, done, info = self.env.step(action)
                    episode_step += 1
                    # update metrics
                    episode_reward += reward
                    # render environment
                    # self.env.render()
                    
                if 'constraint_violation' in info and info['constraint_violation']:
                    cum_constr_viol += 1
                # log metrics to tensorboard
                writerTB.add_scalar("metrics/episode length",
                                  episode_step, eval_step)
                writerTB.add_scalar("metrics/episode reward",
                                  episode_reward, eval_step)
                writerTB.add_scalar("metrics/cumulative constraint violations",
                                  cum_constr_viol, eval_step)
                
                writerTB.add_scalar("metrics/eval guidance replacements", self.eval_guidance_counter, eval_step)
                writerTB.add_scalar("metrics/eval safety replacements", self.eval_safety_counter, eval_step)
                
                data = [eval_step, episode_step, episode_reward, cum_constr_viol, self.eval_guidance_counter, self.eval_safety_counter]
                with open('runs/' + str(self.ts) + '/eval_data.csv', 'a', encoding='UTF8') as f:
                    writer = csv.writer(f)
                    writer.writerow(data)
                    f.close()


                eval_step += 1
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                episode_actions.append(episode_action / episode_step)

            print("Evaluation completed:\n"
                  f"Number of episodes: {len(episode_actions)}\n"
                  f"Average episode length: {np.mean(episode_lengths)}\n"
                  f"Average reward: {np.mean(episode_rewards)}\n"
                  f"Average action magnitude: {np.mean(episode_actions)}\n"
                  f"Cumulative Constraint Violations: {cum_constr_viol}")
            print("----------------------------------------------------------")
        print("==========================================================")
        print(
            f"Finished DDPG training. Time spent: {(time.time() - start_time) // 1} secs")
        print("==========================================================")

        data = [(time.time() - start_time) // 1]
        with open('runs/' + str(self.ts) + '/time.csv', 'a', encoding='UTF8') as f:
            writer = csv.writer(f)
            writer.writerow(data)
            f.close()
        # close environment
        self.env.close()
        # close tensorboard writer
        writerTB.close()


if __name__ == '__main__':

    for _ in range(5):
        Trainer().train()
