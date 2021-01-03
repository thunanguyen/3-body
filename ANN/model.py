import torch
import torch.nn as nn

class ANN(nn.Module):
    def __init__(self):
        super(ANN, self).__init__()
        self.flatten = nn.Flatten()
        self.model = nn.Sequential( nn.ReLU(),
                                    nn.Linear(19,100),
                                    nn.ReLU(),
                                    nn.Linear(100,200),
                                    nn.ReLU(),
                                    nn.Linear(200,400),
                                    nn.ReLU(),
                                    nn.Linear(400,800),
                                    nn.ReLU(),
                                    nn.Linear(800,1600),
                                    nn.ReLU(),
                                    nn.Linear(1600,1600),
                                    nn.ReLU(),
                                    nn.Linear(1600,1600),
                                    nn.ReLU(),
                                    nn.Linear(1600,800),
                                    nn.ReLU(),
                                    nn.Linear(800,400),
                                    nn.ReLU(),
                                    nn.Linear(400,200),
                                    nn.ReLU(),
                                    nn.Linear(200,100),
                                    nn.ReLU(),
                                    nn.Linear(100,18),)
        
    def forward(self, pos, vel, t):
        concat = torch.cat([self.flatten(pos), self.flatten(vel), t], axis=1)
        concat = self.model(concat)
        res = concat.view(concat.size(0), 2, 3, 3)
        
        return res[:, 0, ...], res[:, 1, ...]
                
        
