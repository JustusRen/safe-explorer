import random
import math 
import numpy as np

class Guide:
    def __init__(self, env):
        self._env = env
        self._guided_actions_counter = 0


    def get_guided_actions_counter(self):
        return self._guided_actions_counter
    
    def set_guided_actions_counter(self, guided_actions):
        self._guided_actions_counter = guided_actions
    
    def get_guided_action(self, action, episode):
        # calculate dfa state
        self._env.get_dfa_state(action)
        
        """if self._env.is_guidance_required(): 
            return self._env.get_guided_action(action)
        else:
            return action"""
        #________________________________________________________________________

        #if guidance is required start guiding
        """if self._env.is_guidance_required(): 
            # replace action by a guided action if more than 10 violations in a row (violations are reseted after guidance)
            if self._env.get_guidance_violations() > 10:
                print('guided')
                action = self._env.get_guided_action(action)
            return action"""
          
            
        #________________________________________________________________________

        # replace action based on an increasing probability (violations are reseted after guidance)
        
        random_number = random.randint(0, 25 - self._env.get_guidance_violations())
        if random_number == 0:
            # print('guided')
            action = self._env.get_guided_action(action)
            return action
        
        else:
            return action
        #________________________________________________________________________
        # replace action based on an increasing probability (violations are reseted after guidance)
        
        """step_based_factor = 1 / (1 + np.exp(0.0025 * -episode + 2.5))
        
        scaling_factor = 0.1 * step_based_factor  # Adjust this value to control the rate of increase
        probability = 1 - math.exp(-scaling_factor * guidance_violations)

        # Generate a random number between 0 and 1
        random_number = random.random()

        if random_number <= probability:
            # Condition is satisfied with logarithmic probability
            action = self._env.get_guided_action(action)
            self._guided_actions_counter = self._guided_actions_counter + 1
            return action
        else:
            # Condition is not satisfied
            return action"""
        
        #________________________________________________________________________

        # reset epoch after 10 violations in a row (to-do: reset has to be cleaner)
        """if self._env.get_guidance_violations() > 10:
            print('reset')
            self._env.reset()
            action = [0,0]
            return action"""
        #________________________________________________________________________

        # reset epoch after based on an increasing probability (to-do: reset has to be cleaner)
        """
        random_number = random.randint(0, 10 - guidance_violations)
        if random_number == 0:
            print('reset')
            self._env.reset()
            action = [0,0]
            return action
            """
            
   
