import torch 
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import numpy as np
import  h5py



torch.manual_seed(2)

def load_data(ROOT):
  """ load all of cifar """
  A01T = h5py.File(ROOT, 'r')
  #A01T = h5py.File('A03T_slice.mat', 'r')
  x = np.copy(A01T['image'])
  y = np.copy(A01T['type'])
  y = y[0,0:x.shape[0]:1]
  y = np.asarray(y, dtype=np.int32)
  A01T.close() 
  y[y==769]=0
  y[y==770]=1
  y[y==771]=2
  y[y==772]=3
  A,B,C = x.shape
  a = np.argwhere(np.isnan(x))
  i = 0
  temp = -1
  n_delete = 0
  #jx = x
  #jy = y
  while(i<a[:,0].shape[0]):
      if(a[i,0]!=temp):
          temp = a[i,0]
          x = np.delete(x,temp-n_delete,0)
          y = np.delete(y,temp-n_delete,0)
          n_delete = n_delete+1
      i = i+1    
  
  x = x[:,0:22,:]
  #ja = np.argwhere(np.isnan(x))
  return x, y, n_delete


def get_data(num_training , ROOT):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for classifiers. These are the same steps as we used for the SVM, but
    condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    #ROOT = 'A09T_slice.mat';
    X_train, y_train, n_delete = load_data(ROOT)
        
    # Subsample the data
    mask = range(num_training-n_delete, 288-n_delete)
    X_test = X_train[mask]
    y_test = y_train[mask]

    mask = range(num_training-n_delete)
    X_train = X_train[mask]
    y_train = y_train[mask]
    
    X_valid = X_test[0:38,:,:]
    y_valid = y_test[0:38]
    
    X_test = X_test[38:,:,:]
    y_test = y_test[38:]


#################################################################
    # data augmentation
    X_train1=1/4*X_train
    X_train4=4*X_train
    X_train=np.vstack((X_train,X_train1,X_train4))
    y_train=np.hstack((y_train,y_train,y_train))
    

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_test -= mean_image
    X_train = Variable(torch.Tensor(X_train), requires_grad = False)
    X_test  = Variable(torch.Tensor(X_test ), requires_grad = False)
    y_train = Variable(torch.Tensor(y_train), requires_grad = False)
    y_test  = Variable(torch.Tensor(y_test ), requires_grad = False)
    X_valid = Variable(torch.Tensor(X_valid), requires_grad = False)
    y_valid  = Variable(torch.Tensor(y_valid ), requires_grad = False)
    return X_train, y_train, X_test,  y_test, n_delete, X_valid, y_valid
class Flatten(nn.Module):
    def forward(self, x):
        N,C,H = x.size()
        return x.view(N,-1)


btest_a = [0]*9
bvalid_a = [0]*9
bseed_a = [0]*9
bloss_a = [0]*9


for seeds in range(5):
    torch.manual_seed(seeds)
    #'A01T_slice.mat','A02T_slice.mat','A03T_slice.mat','A04T_slice.mat','A05T_slice.mat','A06T_slice.mat','A07T_slice.mat','A08T_slice.mat','A09T_slice.mat'
    st = ['A01T_slice.mat','A02T_slice.mat','A03T_slice.mat','A04T_slice.mat','A05T_slice.mat','A06T_slice.mat','A07T_slice.mat','A08T_slice.mat','A09T_slice.mat']
    #st = ['A01T_slice.mat']
    
    
    for ss in st:
        num_training=200 
        X_train,y_train,X_test,y_test,n_delete,X_valid, y_valid =  get_data(num_training, ss)
        num_training=200 - n_delete
        dtype = torch.FloatTensor
        N,C,H = 200-n_delete,22,1000
        model = nn.Sequential(
                     nn.Conv1d(C, 22, kernel_size=4, stride=2),
                     nn.ReLU(inplace=True),
                     
                     
                     nn.BatchNorm1d(num_features=22),
                     nn.Dropout(p=0.2, inplace=False),
                     nn.AvgPool1d(kernel_size=8, stride=4),
                     
                     
                     nn.AvgPool1d(kernel_size=4, stride=3),
            
                     
                    ## Reshape(),
             
                     
                     
                     
                     Flatten(),
                     
                     nn.Linear(880, 120),                             
                     #nn.BatchNorm1d(num_features=120),
                     nn.ReLU(inplace=True),
                     nn.Dropout(p=0.60, inplace=False),
                     
                     
                     
                     
                     nn.Linear(120, 4),
                
                          )                                                         
        dtype = torch.FloatTensor                                                                                                        
        model.type(dtype)                                                                                                               
        loss_fn = nn.CrossEntropyLoss().type(dtype)                                                                                                                                                             
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  
        
        
        bestacc=0;
        for t in range(1000):
            model.train()
            y_pred = model(X_train)
            y_pred = y_pred
        
            loss = loss_fn(y_pred, y_train.type(torch.LongTensor))
            #print(loss)
            model.zero_grad()
            loss.backward()
            optimizer.step()
            model.eval()
            y_pred = model(X_valid)
            yy_valid = y_valid.data.numpy()
            yy_pred = y_pred.data.numpy()
            yy_pred = np.argmax(yy_pred,axis = 1)
            valid_acc = np.mean(yy_pred == yy_valid)  
            
            print(valid_acc)  
            print (t)
            print (loss)
            if float(loss) < 0.40:
                indexl = int(ss[2])-1
                if(bvalid_a[indexl]<valid_acc):
                    bloss_a[indexl] = loss
                    bseed_a[indexl] = seeds
                    bvalid_a[indexl] = valid_acc
                    #torch.save(model.state_dict(), ss.replace('.mat',' '))
                    model.eval()
                    y_pred = model(X_test)
                    yy_test = y_test.data.numpy()
                    yy_pred = y_pred.data.numpy()
                    yy_pred = np.argmax(yy_pred,axis = 1)
                    test_acc = np.mean(yy_pred == yy_test)  
                    btest_a[indexl] = test_acc
                
                    print(test_acc)
                    
            if float(loss)< 0.1:
                
                break   
        
    
        
        
        #if(test_acc>bestacc):
        #    bestacc = test_acc
        #    print(test_acc)
            #torch.save(model.state_dict(), ss.replace('.mat',' '))
        
#       y_pred =model(X_train)
#       y_train = y_train.data.numpy()
#       y_pred = y_pred.data.numpy()
#       y_pred = np.argmax(y_pred,axis = 1)
#       train_acc = np.sum(y_pred == y_train)/(num_training)  
    print(ss)
    print(bestacc)
bvalid_a
bloss_a
bseed_a
#%%
print(btest_a)