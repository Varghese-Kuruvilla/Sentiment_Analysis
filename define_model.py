#Script to define the model
from torch import nn,optim
import torch

class Model(nn.Module):

    def __init__(self,words):
        super(Model,self).__init__()

        vocab_size = len(words)
        embedding_dims = 30
        self.embed = nn.Embedding(vocab_size,embedding_dims) #Inputs : vocabulary size and embedding dimensions

        #So for each of the tokenized words we add the dimension 30 to it

        #LSTM Layer
        self.lstm = nn.LSTM(input_size=embedding_dims , hidden_size=512, num_layers=1 , batch_first= True)

        #The output of our LSTM layer is a 50,500,512 tensor which we feed into the FC layer

        self.fc = nn.Linear(in_features = 512, out_features =1)

        #Finally we pass this through a sigmoid layer
        self.sigm = nn.Sigmoid()

    def forward(self,x):

        #print("x.shape:",x.shape)
        
        embed_out = self.embed(x)

        #print("embed_out.shape:",embed_out.shape)
        
        batch_size = 50
        hidden = None
        lstm_out ,h = self.lstm(embed_out,hidden)
        #print("lstm_out:",lstm_out.shape)

        fc_out = self.fc(lstm_out.contiguous().view(-1,512))        
        #print("fc_out:",fc_out.shape)
        
        sigm_out = self.sigm(fc_out)

        #print("sigm_out.shape:",sigm_out.shape) 
        out = sigm_out.view(batch_size,-1)
    
        #print("out[:,-1].view(batch_size,-1).shape",out[:,-1].view(batch_size,-1).shape)
        return out[:,-1].view(batch_size,-1)

def process_model(words,train_loader,test_loader,valid_loader):
    model = Model(words)
    print("model:",model)

    #loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    loss_function = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1,11):
        train_loss , valid_loss = [] , []

        #Training
        model.train()
        for data,target in train_loader:
            print("type(target):",type(target))
            target_flag = 0
            target = target.float()
            print("target.shape[0]:",target.shape[0])
            if(target.shape[0] < 50):
                target_1 = torch.zeros(50)
                target_1[:target.shape[0]] = target
                target = target_1
            
                
                
            optimizer.zero_grad()
            output = model(data)
            print("shape of the output:",output.shape)
            print("target.view(-1,1).shape:",target.view(-1,1).shape)
        
            loss = loss_function(output, target.view(-1,1))
            loss.backward()
            optimizer.step()
            train_loss.append(loss.item())
            print("Inside training loop")

        #Validation
        model.eval()
        for data,target in valid_loader:
            print("Inside evaluation loop:")
            if(target.shape[0] < 50):
                target_1 = torch.zeros(50)
                target_1[:target.shape[0]] = target
                target = target_1

            output = model(data)
            loss = loss_function(output, target.view(-1,1))
            valid_loss.append(loss.item())

        print("Epoch {} , Training_loss{}:".format(epoch , train_loss))
        print("Epoch {} , Validation_loss{}:".format(epoch,valid_loss))

    #dataiter = iter(valid_loader)
    #data , labels = dataiter.next()
    #output = model(data)
    #_,preds_tensor = torch.max(output,1)
    #preds = np.squeeze(preds_tensor.numpy)
