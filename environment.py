# Easy21 Environment
import numpy as np

class Easy21():
    ''' Environment Easy21 '''

    def __init__ (self):
        self.dealer_threshold = 17;
        self.min_card_value, self.max_card_value = 1, 10;
        self.game_lower_bound, self.game_upper_bound = 1, 21;

    def drawCard (self):
        value = np.random.randint(self.min_card_value, self.max_card_value+1)

        if np.random.random() <= 1/3:
            return -value
        else:
            return value
    
    def initGame (self):
        return (np.random.randint(self.min_card_value, self.max_card_value+1),
               np.random.randint(self.min_card_value, self.max_card_value+1))
    
    def actionSpace(self):
        ''' 
        available action for this environment, 0 stands for hit,1 stands for stick
        '''
        return (0,1);
    
    def isBust (self,val):
        if(val < self.game_lower_bound or val > self.game_upper_bound):
            return True;
        return False;

    def dealerTakeCard(self,dealerVal):
        while(dealerVal < self.dealer_threshold and not self.isBust(dealerVal)):
            dealerVal += self.drawCard();
        return dealerVal;
    
    def step (self,dealerVal,playerVal,action):
        '''
        Take step
        Return next state and reward and whether the espisode is terminate

        '''
        terminate = False;
        reward = 0;
        
        # Stick
        if (action == 0):
            playerVal += self.drawCard();
            
            if (self.isBust(playerVal)):
                reward = -1;
                terminate = True;
            
        # Hit
        elif (action == 1):
            terminate = True;
            
            dealerVal = self.dealerTakeCard(dealerVal);
            
            # Check what rewards should get
            if (self.isBust(dealerVal)):
                reward = 1;
            elif (playerVal == dealerVal):
                reward = 0;
            else:
                reward = -1 if (playerVal < dealerVal) else 1;
        
        return dealerVal,playerVal,reward,terminate;
        