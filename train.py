from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np

import utils
import os
import actorcritic as model

BATCH_SIZE = 1024
LEARNING_RATE = 0.01
GAMMA = 0.99
TAU = 0.001


class Trainer:

	def __init__(self, state_dim, action_dim, action_lim, ram, device):
		"""
		:param state_dim: Dimensions of state (int)
		:param action_dim: Dimension of action (int)
		:param action_lim: Used to limit action in [-action_lim,action_lim]
		:param ram: replay memory buffer object
		:param device: device to run the code, 'cpu' or 'cuda' or 'mps
		:return:
		"""
		self.device = torch.device(device)
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.action_lim = torch.tensor(action_lim).float().to(self.device)
		self.ram = ram
		self.iter = 0
		self.noise = utils.OrnsteinUhlenbeckActionNoise(self.action_dim)

		self.actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(self.device)
		self.target_actor = model.Actor(self.state_dim, self.action_dim, self.action_lim).to(self.device)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),LEARNING_RATE)

		self.critic = model.Critic(self.state_dim, self.action_dim).to(self.device)
		self.target_critic = model.Critic(self.state_dim, self.action_dim).to(self.device)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),LEARNING_RATE)

		self.criticlossfunc = nn.SmoothL1Loss()

		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)

		# save losses
		self.actor_loss = []
		self.critic_loss = []

	def get_exploitation_action(self, state):
		"""
		gets the action from target actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state)).float().to(self.device)
		action = self.target_actor.forward(state).detach()
		return action.data.numpy()

	def get_exploration_action(self, state):
		"""
		gets the action from actor added with exploration noise
		:param state: state (Numpy array)
		:return: sampled action (Numpy array)
		"""
		state = Variable(torch.from_numpy(state)).float().to(self.device)
		action = self.actor.forward(state).detach()

		# Ensure the action is within bounds
		# print('new_action', action)

		# for i in range(len(action)):
		# 	assert abs(action[i]) - self.action_lim[i], f'Action {i} is out of bounds'

		lim = self.action_lim.cpu().numpy()
		noise = self.noise.sample() * lim
		new_action = (action.data.cpu().numpy() + noise).clip(-lim, lim)

		# Ensure the action is within bounds
		# print('new_action', new_action)

		# for i in range(len(new_action)):
		# 	assert abs(new_action[i]) <= self.action_lim[i], f'New Action {i} is out of bounds'
		return new_action

	def optimize(self):
		"""
		Samples a random batch from replay memory and performs optimization
		:return:
		"""
		s1,a1,r1,s2 = self.ram.sample(BATCH_SIZE)

		s1 = Variable(torch.from_numpy(s1)).float().to(self.device)
		a1 = Variable(torch.from_numpy(a1)).float().to(self.device)
		r1 = Variable(torch.from_numpy(r1)).float().to(self.device)
		s2 = Variable(torch.from_numpy(s2)).float().to(self.device)

		# ---------------------- optimize critic ----------------------
		# Use target actor exploitation policy here for loss evaluation
		a2 = self.target_actor.forward(s2).detach().float()
		next_val = torch.squeeze(self.target_critic.forward(s2, a2).detach())
		# y_exp = r + gamma*Q'( s2, pi'(s2))
		y_expected = r1 + GAMMA*next_val
		# y_pred = Q( s1, a1)
		y_predicted = torch.squeeze(self.critic.forward(s1, a1))
		# compute critic loss, and update the critic

		if len(y_predicted.shape) == 0:
			y_predicted = torch.unsqueeze(y_predicted, 0)

		loss_critic = self.criticlossfunc(y_predicted, y_expected)
		self.critic_optimizer.zero_grad()
		loss_critic.backward()
		self.critic_optimizer.step()

		self.critic_loss.append(loss_critic.data.cpu().numpy())

		# ---------------------- optimize actor ----------------------
		pred_a1 = self.actor.forward(s1).float()
		loss_actor = -1*torch.sum(self.critic.forward(s1, pred_a1))
		self.actor_optimizer.zero_grad()
		loss_actor.backward()
		self.actor_optimizer.step()

		utils.soft_update(self.target_actor, self.actor, TAU)
		utils.soft_update(self.target_critic, self.critic, TAU)

		self.actor_loss.append(loss_actor.data.cpu().numpy())

		if self.iter % 100 == 0:
			print('Iteration :- ', self.iter, ' Loss_actor :- ', loss_actor.data.cpu().numpy(),\
				' Loss_critic :- ', loss_critic.data.cpu().numpy())
			self.save_losses()
		self.iter += 1

	def save_models(self, episode_count):
		"""
		saves the target actor and critic models
		:param episode_count: the count of episodes iterated
		:return:
		"""
		if not os.path.exists('./Models'):
			os.makedirs('./Models')
		torch.save(self.target_actor.state_dict(), './Models/' + str(episode_count) + '_actor.pt')
		torch.save(self.target_critic.state_dict(), './Models/' + str(episode_count) + '_critic.pt')
		print('Models saved successfully')

	def load_models(self, episode):
		"""
		loads the target actor and critic models, and copies them onto actor and critic models
		:param episode: the count of episodes iterated (used to find the file name)
		:return:
		"""
		self.actor.load_state_dict(torch.load('./Models/' + str(episode) + '_actor.pt'))
		self.critic.load_state_dict(torch.load('./Models/' + str(episode) + '_critic.pt'))
		utils.hard_update(self.target_actor, self.actor)
		utils.hard_update(self.target_critic, self.critic)
		print('Models loaded succesfully')

	def save_losses(self):
		# create a plot of the loss
		import matplotlib.pyplot as plt
		plt.plot(self.actor_loss)
		plt.ylabel('Actor Loss')
		plt.savefig('actor_loss.png')
		plt.clf()

		plt.plot(self.critic_loss)
		plt.ylabel('Critic Loss')
		plt.savefig('critic_loss.png')