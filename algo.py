import numpy as np
from policy import Policy

class ValueFunctionWithApproximation(object):

    def __call__(self,s) -> float:
        """
        return the value of given state; \hat{v}(s)

        input:
            state
        output:
            value of the given state
        """
        raise NotImplementedError()

    def update(self,alpha,G,s_tau):
        """
        Implement the update rule;
        w <- w + \alpha[G- \hat{v}(s_tau;w)] \nabla\hat{v}(s_tau;w)

        input:
            alpha: learning rate
            G: TD-target
            s_tau: target state for updating (yet, update will affect the other states)
        ouptut:
            None
        """
        raise NotImplementedError()

def semi_gradient_n_step_td(
    env, #open-ai environment
    gamma:float,
    pi:Policy,
    n:int,
    alpha:float,
    V:ValueFunctionWithApproximation,
    num_episode:int,
):
    """
    implement n-step semi gradient TD for estimating v

    input:
        env: target environment
        gamma: discounting factor
        pi: target evaluation policy
        n: n-step
        alpha: learning rate
        V: value function
        num_episode: #episodes to iterate
    output:
        None
    """
    #TODO: implement this function
    for episode in range(num_episode):
        state = env.reset()  # Reset environment to initial state
        states = [state]     # To store state trajectory
        rewards = [0]        # To store rewards trajectory (r_0 = 0 for convention)
        T = float('inf')     # Terminal time-step
        t = 0                # Current time-step
        
        while True:
            if t < T:
                # Choose action using policy π
                action = pi.action(state)
                
                # Take action, observe reward and next state
                next_state, reward, done, _ = env.step(action)
                
                # Store reward and state
                rewards.append(reward)
                if not done:
                    states.append(next_state)
                else:
                    T = t + 1  # Mark end of trajectory
            
            # Time of state to update (ensure it's within trajectory)
            tau = t - n + 1
            if tau >= 0:
                # Compute G (n-step return)
                G = 0
                for i in range(tau + 1, min(tau + n + 1, T + 1)):
                    G += gamma ** (i - tau - 1) * rewards[i]
                
                # Add the bootstrapped value if the trajectory is not yet terminal
                if tau + n < T:
                    G += gamma ** n * V(states[tau + n])   
                
                # Update the value function
                V.update(alpha, G, states[tau])
            
            if tau == T - 1:
                break
            
            t += 1
            if t < T:
                state = next_state
