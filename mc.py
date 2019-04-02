# MC Learning
from environment import Easy21
import numpy as np
import utils
import dill as pickle

class MC():

    def __init__(self):
        self.Q = None;

    def train(self,episodes = 100000,printEvery = 1000,N0 = 100):
        def epilsonGreedy(dealerVal,playerVal):
            ''' Return the action that should make'''
            if( np.random.random() < epilson(dealerVal,playerVal)):
                action = np.random.choice(env.actionSpace());
            else:
                action = np.argmax( [Q[dealerVal,playerVal,action] for action in env.actionSpace()] )
            return action;
        # Create Environment
        env = Easy21();

        # Q-Function - State-Action function
        Q = np.zeros((11,22,len(env.actionSpace())));

        # Number of State, Action - Record number of times that action a selected from state s
        NSA = np.zeros((11,22,len(env.actionSpace())));

        # Number of time state s being visited
        NSV = lambda d,p: sum(NSA[d,p]);

        # Step-size
        LearningRate = lambda d,p,a: 1 / NSA[d,p,a];

        # Epilson greedy function
        epilson = lambda d,p: N0 / (N0 + NSV(d,p));

        meanReturn = 0;
        wins = 0;

        # Start Training
        for i in range(episodes+1):
            terminate = False;
            
            # To store State Action Reward
            SAR = []
            dealer, player = env.initGame();

            while not terminate:
                # Use policy to choose action
                action = epilsonGreedy(dealer,player);

                # Record the action from state
                NSA[dealer,player,action] += 1;

                # dealerPrime, playerPrime, reward, terminate = env.step(dealer,player,action);
                dealerPrime, playerPrime, reward, terminate = env.step(dealer,player,action);

                # Save State Action Reward
                SAR.append([dealer,player,action,reward]);
                
                # Move to next state
                dealer, player = dealerPrime, playerPrime

            # Total rewards so far
            G = sum([sar[-1] for sar in SAR]);

            # Update Q function
            for(dealer,player,action,_) in SAR:
                Q[dealer,player,action] += LearningRate(dealer,player,action) * (G - Q[dealer,player,action]);

            # Showing result
            meanReturn += (G-meanReturn)/(i+1);
            if(reward == 1):
                wins += 1;
            
            if(i%printEvery==0):
                print("Episode %i, Mean-Return %.3f, Wins %.2f"%(i, meanReturn, (wins)/(i+1)))

        self.Q = Q;
         
    def draw(self):
        # Save Q in dill for later purpose
        pickle.dump(self.Q, open('Q.dill', 'wb'))
        utils.plot(self.Q, [0,1])


MCLearning = MC();
MCLearning.train();
MCLearning.draw();
