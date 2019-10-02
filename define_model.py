#Script to define the model
from torch import nn


class Model(nn.Module):

    def__init__(self):
        super(Model,self).__init__()

        vocab_size = len(self.words)
        embedding_dims = 30
        self.embed = nn.Embedding(vocab_size,embedding_dims) #Inputs : vocabulary size and embedding dimensions

        #So for each of the tokenized words we add the dimension 30 to it

        #LSTM Layer
        hidden = None
        self.lstm = nn.LSTM(input_size=embedding_dims , hidden_size=512, num_layers=1 , batch_first= True)

        #The output of our LSTM layer is a 50,500,512 tensor which we feed into the FC layer

        self.fc = nn.Linear(in_features = 512, out_features =1 )

        #Finally we pass this through a sigmoid layer
        self.sigm = nn.Sigmoid()

        #Finally reshape the output so that the number of rows = batch size
        batch_size = self.x.shape[0]
        out = sigm_out.view(batch_size,-1)
        print("Shape of the final output:",out.shape)

    def forward(self,x):
        
        embed_out = self.embed(self.x)
         
        lstm_out ,h = self.lstm(embed_out,hidden)

        fc_out = self.fc(lstm_out.contiguous().view(-1,512))
        
        sigm_out = self.sigm(fc_out)
        
        out = sigm_out.view(batch_size,-1)

        return out

def process_model(words,train_loader,test_loader,valid_loader):
    model = Model()

    loss_function = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1,11):
        train_loss , valid_loss = [] , []

    #Training
    model.train()
    for data,target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = loss_function(output, target.view(-1,1))
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    #Validation
    model.eval()
    for data,target in valid_loader:
        output = model(data)
        loss = loss_function(output, target.view(-1,1))
        valid_loss.append(loss.item())
