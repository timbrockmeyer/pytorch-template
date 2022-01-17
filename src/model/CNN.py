from torch import nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(         
            nn.Conv2d(
                in_channels=1,              
                out_channels=16,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                in_channels=16,              
                out_channels=32,            
                kernel_size=5,              
                stride=1,                   
                padding=2,                  
            ),                              
            nn.ReLU(),                      
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(32 * 7 * 7, 10) 
        )
        
    def forward(self, x):
        out = self.model(x)

        return out    
        