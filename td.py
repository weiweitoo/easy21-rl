# SARSA
from environment import Easy21
import numpy as np
import utils
import dill as pickle

class SARSA():

	def __init__(self):
		self.finalMSE = None;
		self.lambdaRange = None;
		self.mseLambda = None;

	def train(self,episodes = 10000,printEvery = 1000,N0 = 100):
		def reset():
		   Q = np.zeros((11,22,len(env.actionSpace())));
		   NSA = np.zeros((11,22,len(env.actionSpace())));
		   wins = 0;
		   return (Q,NSA,wins);

		def epilsonGreedy(dealerVal,playerVal):
			if( np.random.random() < epilson(dealerVal,playerVal)):
				action = np.random.choice(env.actionSpace());
			else:
				action = np.argmax( [Q[dealerVal,playerVal,action] for action in env.actionSpace()] )
			return action;

		# Create Environment
		env = Easy21();

		Q, NSA, wins = reset();
		trueQ = pickle.load(open('Q.dill', 'rb'))

		# Number of time state s being visited
		NSV = lambda d,p: sum(NSA[d,p]);

		# Step-size
		learningRate = lambda d,p,a: 1 / NSA[d,p,a];

		# Range of lambda that will try
		lambdaRange = list(np.arange(0,11)/10);

		# MSE for each lambda of each episode
		mseLambda = np.zeros((len(lambdaRange),episodes));

		# The final MSE of each lambda
		finalMSE = np.zeros((len(lambdaRange)));

		# Epilson greedy function
		epilson = lambda d,p: N0 / (N0 + NSV(d,p));

		# discount factor
		gamma = 1;

		# Start Training
		for i,lmd in enumerate(lambdaRange):

			# Reset for next lambda
			Q, NSA, wins = reset();

			for episode in range(episodes):
				terminate = False;

				# Eligibility trace
				E = np.zeros((11,22,len(env.actionSpace())));
				
				# To store State Action
				SA = []

				dealer, player = env.initGame();

				# Use policy to choose first action
				action = epilsonGreedy(dealer,player);

				while not terminate:

					# Record the action from state
					NSA[dealer,player,action] += 1;

					# Record for Egibility Trace
					E[dealer,player,action] += 1;

					# Save State Action
					SA.append([dealer,player,action]);

					dealerPrime, playerPrime, reward, terminate = env.step(dealer,player,action);

					if not terminate:
						actionPrime = epilsonGreedy(dealerPrime,playerPrime);
						TDTarget = reward + Q[dealerPrime,playerPrime,actionPrime];
					else:
						TDTarget = reward;

					# Update Q function
					for (_dealer,_player,_action) in SA:
						Q[_dealer,_player,_action] += learningRate(_dealer,_player,_action) * (TDTarget - Q[dealer,player,action]) * E[_dealer,_player,_action]
						E[_dealer,_player,_action] *= lmd * gamma;

					# Move to next state for next move
					if not terminate:
						dealer, player, action = dealerPrime, playerPrime, actionPrime;
			   
				if(reward == 1):
					wins +=1;

				mse = np.sum(np.square(Q-trueQ)) / (21*10*2);
				mseLambda[i,episode] = mse;

				if episode % printEvery == 0 or episode+1==episodes:
					print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f"%(lmd, episode, mse, wins/(episode+1)))

			finalMSE[i] = mse
			print("Lambda=%.1f Episode %06d, MSE %5.3f, Wins %.3f \n"%(lmd, episode, mse, wins/(episode+1)))

		self.finalMSE = finalMSE;
		self.lambdaRange = lambdaRange;
		self.mseLambda = mseLambda;
		 
	def draw(self):
		utils.plotMseLambdas(self.finalMSE, self.lambdaRange)
		utils.plotMseEpisodesLambdas(self.mseLambda)


SARSALearning = SARSA();
SARSALearning.train();
SARSALearning.draw();