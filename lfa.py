# LFA
from environment import Easy21
import numpy as np
import utils
import dill as pickle

class LFA():

    def __init__(self):
        self.finalMSE = None;
        self.lambdaRange = None;
        self.mseLambda = None;

    def train(self,episodes = 10000,printEvery = 1000):

        # Create Environment
        env = Easy21();

        trueQ = pickle.load(open('Q.dill', 'rb'))

        # Number of time state s being visited
        NSV = lambda d,p: sum(NSA[d,p]);

        # Step-size
        learningRate = 0.01;

        # Range of lambda that will try
        lambdaRange = list(np.arange(0,11)/10);

        # MSE for each lambda of each episode
        mseLambda = np.zeros((len(lambdaRange),episodes));

        # The final MSE of each lambda
        finalMSE = np.zeros((len(lambdaRange)));

        # Epilson greedy function
        epilson = 0.05;

        # discount factor
        gamma = 1;

        # features
        dealerFeatures = [[1, 4], [4, 7], [7, 10]]
        playerFeatures = [[1, 6], [4, 9], [7, 12], [10, 15], [13, 18], [16, 21]]
        allFeatures = np.zeros((11, 22, 2, 3*6*2))

        def reset():
            theta = np.random.randn(3*6*2, 1);
            wins = 0;
            return theta, wins

        def epilsonGreedy(dealerVal,playerVal):
            if( np.random.random() < epilson):
                action = np.random.choice(env.actionSpace());
            else:
                action = np.argmax( [Q(dealerVal,playerVal,action) for action in env.actionSpace()] )
            return action;

        def features(dealer, player, action):
            f = np.zeros(3*6*2);

            for fi, (lower, upper) in enumerate(dealerFeatures):
                f[fi] = (lower <= dealer <= upper);

            for fi, (lower, upper) in enumerate(playerFeatures, start=3):
                f[fi] = (lower <= player <= upper);

            f[-2] = 1 if action == 0 else 0;
            f[-1] = 1 if action == 1 else 0;

            return f.reshape(1, -1);

        def Q(dealer, player, action):
            return np.dot(features(dealer, player, action), theta)

        def allQ():
            return np.dot(allFeatures.reshape(-1, 3*6*2), theta).reshape(-1)

        # Init feature
        for d in range(1, 11):
            for p in range(1, 22):
                for a in range(0, 2):
                    allFeatures[d-1, p-1, a] = features(d, p, a)

        # Start Training
        for i,lmd in enumerate(lambdaRange):

            # Reset for next lambda
            theta, wins = reset()

            for episode in range(episodes):

                terminate = False;

                # Eligibility trace
                E = np.random.randn(3*6*2, 1);

                dealer, player = env.initGame();

                # Use policy to choose first action
                action = epilsonGreedy(dealer,player);

                while not terminate:

                    dealerPrime, playerPrime, reward, terminate = env.step(dealer,player,action);

                    if not terminate:
                        actionPrime = epilsonGreedy(dealerPrime,playerPrime);
                        TDTarget = reward + Q(dealerPrime,playerPrime,actionPrime);
                    else:
                        TDTarget = reward;

                    E = lmd * E + features(dealer, player, action).reshape(-1, 1);
                    gradient = learningRate * (TDTarget - Q(dealer,player,action)) * E;
                    theta = theta + gradient;

                    # Move to next state for next move
                    if not terminate:
                        dealer, player, action = dealerPrime, playerPrime, actionPrime;
               
                if(reward == 1):
                    wins +=1;

                mse = np.sum(np.square(allQ() - trueQ.ravel())) / (21*10*2)
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

LFALearning = LFA();
LFALearning.train();
LFALearning.draw();