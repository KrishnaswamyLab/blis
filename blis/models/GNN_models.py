import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, GATConv, ChebConv
from torch_geometric.nn import GINConv, global_mean_pool
from torch.nn import Sequential, Linear, ReLU
from blis.models.spectral_conv import SpectConv, ML3Layer
import torch

class GCN(nn.Module):
    def __init__(self, in_features,hidden_channels, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim = 1)
        self.in_features = in_features
    
    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        
        if len(x.shape) == 1:
            x = x[:,None]
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        #x = F.dropout(x, p = 0.2, training=self.training)
        x = self.lin(x)
        # Apparently when using cross entropy loss, the softmax is automatically applied.
        #x = self.final_nonlin(x)

        return x

class GIN(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes):
        super(GIN, self).__init__()

        # Define the neural network for GINConv (this can be adjusted)
        nn1 = Sequential(Linear(in_features, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv1 = GINConv(nn1)
        
        nn2 = Sequential(Linear(hidden_channels, hidden_channels), ReLU(), Linear(hidden_channels, hidden_channels))
        self.conv2 = GINConv(nn2)
        
        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim=1)
        self.in_features = in_features

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch

        if len(x.shape) == 1:
            x = x[:,None]
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x

class GAT(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes, heads=1):
        super(GAT, self).__init__()

        # The number of heads can be adjusted for multi-head attention.
        self.conv1 = GATConv(in_features, hidden_channels, heads=heads, concat=True)
        # If we use multi-head attention in the first layer, we need to adjust the input dimension of the next layer.
        self.conv2 = GATConv(heads * hidden_channels, hidden_channels)

        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim=1)
        self.in_features = 1

    def forward(self, data):
        x = data.x
        edge_index = data.edge_index
        batch = data.batch
        
        if len(x.shape) == 1:
            x = x[:,None]
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = F.elu(x)

        x = global_mean_pool(x, batch)

        x = self.lin(x)

        return x
    
class ChebNet(nn.Module):
    def __init__(self,in_features, hidden_channels, num_classes, S=5):
        super(ChebNet, self).__init__()

        #self.conv1 = ChebConv(in_features, 32,S)
        #self.conv2 = ChebConv(32, hidden_channels, S)
        #self.conv3 = ChebConv(hidden_channels, hidden_channels, S)        
        #self.fc1 = torch.nn.Linear(hidden_channels, 10)
        #self.fc2 = torch.nn.Linear(10, num_classes)
        self.conv1 = ChebConv(in_features, hidden_channels, S)
        self.conv2 = ChebConv(hidden_channels, hidden_channels, S)
        self.lin = Linear(hidden_channels, num_classes)
        self.final_nonlin = nn.Softmax(dim=1)
        
    def forward(self, data):
        x=data.x

        if len(x.shape)==1:
            x=x[:,None]
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index,lambda_max=data.lambda_max,batch=data.batch))              
        x = F.relu(self.conv2(x, edge_index,lambda_max=data.lambda_max,batch=data.batch))        
        #x = F.relu(self.conv3(x, edge_index,lambda_max=data.lambda_max,batch=data.batch))
        x = global_mean_pool(x, data.batch)
        #x = F.relu(self.fc1(x))
        x = self.lin(x)
        return x

    
class GNNML1(nn.Module):
    def __init__(self, in_features, hidden_channels = 64, num_classes = 1):
        super(GNNML1, self).__init__()
        
        # number of neuron
        nout=hidden_channels   
        # three part concatenate or sum?
        self.concat=False

        if self.concat:
            nin=3*nout
        else:
            nin=nout
        self.conv11 = SpectConv(in_features, nout,selfconn=False)
        self.conv21 = SpectConv(nin, nout, selfconn=False)
        self.conv31 = SpectConv(nin, nout, selfconn=False)
        
        self.fc11 = torch.nn.Linear(in_features, nout)
        self.fc21 = torch.nn.Linear(nin, nout)
        self.fc31 = torch.nn.Linear(nin, nout)
        
        self.fc12 = torch.nn.Linear(in_features, nout)
        self.fc22 = torch.nn.Linear(nin, nout)
        self.fc32 = torch.nn.Linear(nin, nout)

        self.fc13 = torch.nn.Linear(in_features, nout)
        self.fc23 = torch.nn.Linear(nin, nout)
        self.fc33 = torch.nn.Linear(nin, nout)
        
        self.fc1 = torch.nn.Linear(nin, 10)
        self.fc2 = torch.nn.Linear(10, num_classes)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):

        x=data.x

        if len(x.shape)==1:
            x=x[:,None]
              
        edge_index=data.edge_index
        # need to send this to device!
        edge_attr=torch.ones(edge_index.shape[1],1).to(self.device)
        
        if self.concat:
            x = torch.cat([F.relu(self.fc11(x)), F.relu(self.conv11(x, edge_index,edge_attr)),F.relu(self.fc12(x)*self.fc13(x))],1)
            x = torch.cat([F.relu(self.fc21(x)), F.relu(self.conv21(x, edge_index,edge_attr)),F.relu(self.fc22(x)*self.fc23(x))],1)
            x = torch.cat([F.relu(self.fc31(x)), F.relu(self.conv31(x, edge_index,edge_attr)),F.relu(self.fc32(x)*self.fc33(x))],1)
        else:
            x = F.relu(self.fc11(x)+self.conv11(x, edge_index,edge_attr)+self.fc12(x)*self.fc13(x))
            x = F.relu(self.fc21(x)+self.conv21(x, edge_index,edge_attr)+self.fc22(x)*self.fc23(x))
            x = F.relu(self.fc31(x)+self.conv31(x, edge_index,edge_attr)+self.fc32(x)*self.fc33(x))
        

        x = global_mean_pool(x, data.batch)
        x = self.fc1(x)
        return self.fc2(x)

class GNNML3(nn.Module):
    def __init__(self, in_features, n_edges = 64, num_classes =1):
        super(GNNML3, self).__init__()

        # number of neuron for for part1 and part2
        nout1=16
        nout2=8

        nin=nout1+nout2
        ne= n_edges
        ninp=in_features

        self.conv1=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2) 
        
        self.fc1 = torch.nn.Linear(nin, 10)
        self.fc2 = torch.nn.Linear(10, num_classes)
        

    def forward(self, data):
        x=data.x

        if len(x.shape)==1:
            x=x[:,None]
        
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2

        x=(self.conv1(x, edge_index,edge_attr))
        x=(self.conv2(x, edge_index,edge_attr))
        x=(self.conv3(x, edge_index,edge_attr))  

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MLP(nn.Module):
    def __init__(self, in_features, hidden_channels, num_classes):
        super(MLP, self).__init__() 

        # define some MLP layers, ignoring the graph structure 
        self.lin1 = Linear(in_features, hidden_channels)
        self.lin2 = Linear(hidden_channels, num_classes)
        self.nonlin = nn.ReLU()

    def forward(self, data):
        x = data.x 
        x = self.lin1(x)
        x = self.nonlin(x)
        x = self.lin2(x)
        return x
    

class PPGN(nn.Module):
    def __init__(self,in_features, hidden_channels, num_classes = 1):
        super(PPGN, self).__init__()

        #self.nmax=nmax        
        self.nneuron= hidden_channels
        ninp=in_features
        
        bias=True
        self.mlp1_1 = torch.nn.Conv2d(ninp,self.nneuron,1,bias=bias) 
        self.mlp1_2 = torch.nn.Conv2d(ninp,self.nneuron,1,bias=bias) 
        self.mlp1_3 = torch.nn.Conv2d(self.nneuron+ninp, self.nneuron,1,bias=bias) 

        self.mlp2_1 = torch.nn.Conv2d(self.nneuron,self.nneuron,1,bias=bias) 
        self.mlp2_2 = torch.nn.Conv2d(self.nneuron,self.nneuron,1,bias=bias) 
        self.mlp2_3 = torch.nn.Conv2d(2*self.nneuron,self.nneuron,1,bias=bias) 

        self.mlp3_1 = torch.nn.Conv2d(self.nneuron,self.nneuron,1,bias=bias) 
        self.mlp3_2 = torch.nn.Conv2d(self.nneuron,self.nneuron,1,bias=bias) 
        self.mlp3_3 = torch.nn.Conv2d(2*self.nneuron,self.nneuron,1,bias=bias) 

        self.mlp4_1 = torch.nn.Conv2d(self.nneuron,self.nneuron,1,bias=bias) 
        self.mlp4_2 = torch.nn.Conv2d(self.nneuron,self.nneuron,1,bias=bias) 
        self.mlp4_3 = torch.nn.Conv2d(2*self.nneuron,self.nneuron,1,bias=bias) 

        self.h1 = torch.nn.Linear(1*4*self.nneuron, num_classes)        


    def forward(self,data):

        x=data.X2 
        M=torch.sum(data.M,(1),True)              

        x1=F.relu(self.mlp1_1(x)*M) 
        x2=F.relu(self.mlp1_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp1_3(torch.cat([x1x2,x],1))*M)         

        # read out mean or add ?        
        xo1=torch.sum(x*data.M[:,0:1,:,:],(2,3))

        x1=F.relu(self.mlp2_1(x)*M) 
        x2=F.relu(self.mlp2_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp2_3(torch.cat([x1x2,x],1))*M) 

        # read out         
        xo2=torch.sum(x*data.M[:,0:1,:,:],(2,3))


        x1=F.relu(self.mlp3_1(x)*M) 
        x2=F.relu(self.mlp3_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp3_3(torch.cat([x1x2,x],1))*M) 

        # read out         
        xo3=torch.sum(x*data.M[:,0:1,:,:],(2,3))

        x1=F.relu(self.mlp4_1(x)*M) 
        x2=F.relu(self.mlp4_2(x)*M)  
        x1x2 = torch.matmul(x1, x2)*M
        x=F.relu(self.mlp4_3(torch.cat([x1x2,x],1))*M) 
        
        # read out        
        xo4=torch.sum(x*data.M[:,0:1,:,:],(2,3))
        
        x=torch.cat([xo1,xo2,xo3,xo4],1)         
        
        return F.log_softmax(self.h1(x), dim=1)
