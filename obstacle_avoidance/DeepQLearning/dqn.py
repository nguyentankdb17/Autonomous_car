import torch
from torch import nn
import torch.nn.functional as F

class DQN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim) 
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)  
        self.fc3 = nn.Linear(hidden_dim, output_dim)  

    def forward(self, x):
        x = F.relu(self.fc1(x))      
        x = F.relu(self.fc2(x))      
        x = self.fc3(x)                
        return x
    
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DQN(input_dim=12, hidden_dim=128, output_dim=2).to(device)
    # Additional code for training or evaluating the model can be added here

if __name__ == "__main__":
    main()
