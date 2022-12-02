#FeedForward Neural Network
class Net(nn.Module):
    def __init__(self, K_Length, num_units, activation, dropout, usebias):
        super(Net, self).__init__()
        self.K_Length = K_Length
        self.num_units = num_units
        self.activation = activation
        self.dropout = dropout
        self.usebias = usebias        
        
        self.layers = nn.ModuleList([
            nn.Linear(in_features = self.num_units[0], out_features = self.num_units[1], bias = self.usebias[0]),
            nn.Dropout(p = self.dropout[0], inplace = False),
            self.GetActivationLayer(0),
            nn.Linear(in_features = self.num_units[1], out_features = self.num_units[2], bias = self.usebias[1]),
            nn.Dropout(p = self.dropout[1], inplace = False),
            self.GetActivationLayer(1),
            nn.Linear(in_features = self.num_units[2], out_features = self.num_units[3], bias = self.usebias[2]),
            nn.Dropout(p = self.dropout[2], inplace = False),
            self.GetActivationLayer(2),
            nn.Linear(in_features = self.num_units[3], out_features = self.K_Length, bias = self.usebias[3])
        ])
    
    def forward(self, x):
        output = x.view(x.size(0), -1)
        output = self.layers[0](output)
        output = self.layers[1](output)
        output = self.layers[2](output)
        output = self.layers[3](output)
        output = self.layers[4](output)
        output = self.layers[5](output)
        output = self.layers[6](output)
        output = self.layers[7](output)
        output = self.layers[8](output)
        output = self.layers[9](output)
        return output
    
    def GetActivationLayer(self, layer):
        Result = None
        if (self.activation[layer] == "relu"): #Not differentiable at 0. Doesn't need Greedy layer-wise pretraining (Hinton) because it doesn't suffer from vanishing gradient
            Result = nn.LeakyReLU(ReluAlpha) if ReluAlpha != 0 else nn.ReLU() #alpha: Controls the angle of the negative slope
        elif (self.activation[layer] == "relu6"):
            Result = nn.ReLU6()
        elif (self.activation[layer] == "elu"): #Like ReLu but allows values to be negative, so they can be centred around 0, also potential vanishing gradient on the left side but doesn't matter
            Result = nn.ELU(alpha = EluAlpha) #alpha: Slope on the left side
        elif (self.activation[layer] == "tanh"): #Suffers from Vanishing Gradient
            Result = nn.Tanh()
        elif (self.activation[layer] == "sigmoid"): #Suffers from Vanishing Gradient
            Result = nn.Sigmoid() #Result isn't centred around 0. Maximum derivative: 0.25
        return Result
print("Done")