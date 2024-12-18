import numpy as np
from algo import ValueFunctionWithApproximation
import torch
import torch.nn as nn
import torch.optim as optim

class ValueFunctionWithNN(ValueFunctionWithApproximation):
    def __init__(self,
                 state_dims):
        """
        state_dims: the number of dimensions of state space
        """
        # TODO: implement this method
        super().__init__()
        # Define the neural network structure
        self.model = nn.Sequential(
            nn.Linear(state_dims, 32),  # Input layer to 1st hidden layer
            nn.ReLU(),                 # Activation function
            nn.Linear(32, 32),         # 1st hidden layer to 2nd hidden layer
            nn.ReLU(),                 # Activation function
            nn.Linear(32, 32),         # 2nd hidden layer to 3rd hidden layer
            nn.ReLU(),                 # Activation function
            nn.Linear(32, 1)           # 3rd hidden layer to output layer
        )
        
        # Use Adam optimizer with specified parameters
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, betas=(0.9, 0.999))
        
        # Use Mean Squared Error (MSE) as the loss function
        self.criterion = nn.MSELoss()


    def __call__(self,s):
        # TODO: implement this method
        self.model.eval()  # Set model to evaluation mode
        
        # Convert state `s` to a PyTorch tensor (ensure it is float)
        s_tensor = torch.tensor(s, dtype=torch.float32)
        
        # If `s` is a single state, add a batch dimension
        if s_tensor.ndim == 1:
            s_tensor = s_tensor.unsqueeze(0)
        
        # Forward pass through the model
        with torch.no_grad():  # No gradient computation
            value = self.model(s_tensor).item()  # Get scalar output
        
        return value

    def update(self,alpha,G,s_tau):
        # TODO: implement this method

        self.model.train()  # Set model to training mode
        
        # Convert `s_tau` to tensor and ensure it's float
        s_tau_tensor = torch.tensor(s_tau, dtype=torch.float32)
        
        if s_tau_tensor.ndim == 1:
            s_tau_tensor = s_tau_tensor.unsqueeze(0)  # Add batch dimension
        

        
        # Compute the target value `G`
        target_value = torch.tensor( G, dtype=torch.float32)  # Target value
        

        
        # Perform backpropagation and update weights
        self.optimizer.zero_grad()  # Clear gradients
        # Compute the predicted value for `s_tau`
        predicted_value = self.model(s_tau_tensor)

        # Compute the loss
        loss = self.criterion(predicted_value, target_value)
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update weights

        return None
