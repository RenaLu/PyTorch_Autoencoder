class Encoder(nn.Module):
    def __init__(self, sizes):
        super(Encoder, self).__init__()
                
        layers_en = OrderedDict()       
        for i in range(len(sizes)-1):
            layer_name = 'linear{}'.format(i+1)
            act_name = 'activation{}'.format(i+1)
            layers_en[layer_name] = nn.Linear(sizes[i], sizes[i+1])
            if i==0:
                nn.init.xavier_uniform_(layers_en[layer_name].weight)
            layers_en[act_name] = nn.ReLU()
        
        self.encoder = nn.Sequential(layers_en)

    def forward(self, x):
        return self.encoder(x) 
    
class Decoder(nn.Module):
    def __init__(self, sizes):
        super(Decoder, self).__init__()
        
        sizes = sizes[::-1]
        
        layers_de = OrderedDict()
        for i in range(len(sizes)-2):
            layer_name = 'linear{}'.format(i+1)
            act_name = 'activation{}'.format(i+1)
            layers_de[layer_name] = nn.Linear(sizes[i], sizes[i+1])
            layers_de[act_name] = nn.ReLU()

        layers_de['linear{}'.format(len(sizes)-1)] = nn.Linear(sizes[-2], sizes[-1])
        layers_de['sigmoid'] = nn.Sigmoid()
        self.decoder = nn.Sequential(layers_de)

    def forward(self, encoded):
        return self.decoder(encoded)

# Can be customized according to need (i.e. combining loss functions or change weights
def loss_func(data, decoded):
    cossim_loss = nn.CosineEmbeddingLoss() # Pytorch built-in Cosine similarity for calculating loss 
    y = torch.tensor(np.ones((data.shape[0], 1)), dtype=torch.float).cuda()
    mse_loss = nn.MSELoss()
    loss = cossim_loss(data, decoded, y)
            
    return loss

def train(data, encoder, decoder, name, lr=0.01, iterations=2000):
    data = data.type(torch.FloatTensor)

    if use_cuda and torch.cuda.is_available():
        data = data.cuda()

    optimizer_en = optim.Adam(encoder.parameters(), lr=lr)
    scheduler_en = optim.lr_scheduler.ReduceLROnPlateau(optimizer_en, 'min', patience=1000)
    optimizer_de = optim.Adam(decoder.parameters(), lr=lr)
    scheduler_de = optim.lr_scheduler.ReduceLROnPlateau(optimizer_de, 'min', patience=1000)
        
    n = data.shape[0]

    for it in range(iterations):
        total_loss = 0
        for row_num in range(n):
        row = data[[row_num]]

        optimizer_en.zero_grad()
        optimizer_de.zero_grad()

        encoded = encoder(row)
        decoded = decoder(encoded)

        loss = loss_func(row, decoded)              
        loss.backward()

        optimizer_en.step()        
        optimizer_de.step()    

        total_loss += loss
                  
        scheduler_en.step(total_loss)
        scheduler_de.step(total_loss)
                
        if (it+1)%50 == 0:
            print("Iteration: ({}/{}) Total Loss: {} LR_encoder:{} LR_decoder:{}".format(
                it+1, iterations, total_loss, optimizer_en.param_groups[0]['lr'], optimizer_de.param_groups[0]['lr']))
    
