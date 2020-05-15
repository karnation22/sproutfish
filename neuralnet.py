import torch 
import torch.nn as nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torch.utils.data import Dataset, DataLoader
from torch import tensor
import pandas as pd, numpy as np
import math

#### RELEVANT MACROS ##########
filename1 = 'Jan06 week3.xls'
filename2 = 'Jan06 week4.xls' 
filename3 = 'Grouper production feed and FCR forecast.xlsx'
model_weights = 'fish_wghts.pth'
INPUT_DIM = 6
HIDDEN_DIM_1,HIDDEN_DIM_2 = 8,4
OUTPUT_DIM = 2
BATCH_SIZE = 10
NUM_EPOCH = 10
L_RATE = 0.01
MOMENTUM = 0.9
prm_str='Parameters'
main_col = 'Aquarium Water Chemistry'
lst_row = 'Clione Lab'
acc_lm_str = 'Acceptable limits'
rel_cols = ['Salinity', 'pH', 'Temp', 'DO (mg/L)', 'NH3', 'NO2-'] ##relevant tunable parameters...
rel_cols_opt = ['Total biomass, kg',"% Growth/day", 'Monthly feed consumption, kg']

##############################
### HARDCODE BOUNDS ###
sal_lw,sal_hi=29.0,35.0
sal2_lw,sal2_hi=0.0,3.0
ph_lw,ph_hi=7.5,8.3
tmp_lw,tmp_hi=0.0,40.0
do_lw,do_hi=4.0,10.0
nh3_lw,nh3_hi=0,0.10
no_lw,no_hi=0,0.15
rel_bounds = [((sal_lw,sal_hi),(sal2_lw,sal2_hi)),(ph_lw,ph_hi),
    (tmp_lw,tmp_hi),(do_lw,do_hi),(nh3_lw,nh3_hi),(no_lw,no_hi)]
F_NORM = lambda x, minn, maxx: (x-minn)/(maxx-minn)
#######################

## normalize columns for input parameters
def normalize_cols(pd_inp):
    for col in pd_inp:
        if(col in rel_cols):
            rel_ind = rel_cols.index(col)
            rel_bnd = rel_bounds[rel_ind]   
            if(col=='Salinity'):
                (lw1,up1),(lw2,up2) = rel_bnd
                pd_inp[col] = pd_inp[col].replace('30..9','30.9').replace('n.a','NaN').replace('Tank Closed','NaN') ## minor typo + NaNify everything...
                pd_inp.loc[(pd_inp[col].astype(float)>=lw1)&(pd_inp[col].astype(float)<=up1),col]=(pd_inp.loc[(pd_inp[col].astype(float)>=lw1)&(pd_inp[col].astype(float)<=up1),col].astype(float)-lw1)/(up1-lw1)
                pd_inp.loc[(pd_inp[col].astype(float)>=lw2)&(pd_inp[col].astype(float)<=up2),col]=(pd_inp.loc[(pd_inp[col].astype(float)>=lw2)&(pd_inp[col].astype(float)<=up2),col].astype(float)-lw2)/(up2-lw2)
                
            else:
                (lw,up)=rel_bnd
                pd_inp[col] = pd_inp[col].replace('n.a','NaN').replace('Tank Closed','NaN')
                if(col=='NO2-'): pd_inp[col]=pd_inp[col].replace('-',0.5*(no_hi-no_lw))
                #print(pd_inp[col])
                pd_inp[col]=(pd_inp[col].astype(float)-lw)/(up-lw)
 
    return pd_inp[rel_cols].dropna()


def pd_pre_proc():
    pd_w3 = pd.read_excel(filename1).iloc[3:]
    pd_w3.rename(columns=pd_w3.iloc[0],inplace=True)
    pd_w3 = pd_w3[1+5:-11] #remove header row + 5 more other stray rows... remove last eleven non-data rows
    pd_w3_nrm = normalize_cols(pd_w3)
    pd_w4 = pd.read_excel(filename2).iloc[3:]
    pd_w4.rename(columns=pd_w4.iloc[0],inplace=True)
    pd_w4 = pd_w4[1+5:-11] #remove header row + 5 more other stray rows... remove last elevent non-data rows
    pd_w4_nrm = normalize_cols(pd_w4)
    pd_opt = pd.read_excel(filename3).iloc[3:]#.dropna() #remove first three rows...
    pd_opt_1a = pd_opt.iloc[:,0]
    pd_opt_1b = pd_opt.iloc[:,1:7]
    pd_opt_2a = pd_opt.iloc[:,0] 
    pd_opt_2b = pd_opt.iloc[:,10:16]
    pd_opt_1 = pd.concat([pd_opt_1a,pd_opt_1b],axis=1).dropna(how='all').transpose()
    pd_opt_2  = pd.concat([pd_opt_2a,pd_opt_2b],axis=1).dropna(how='all').transpose()
    pd_opt_1.columns, pd_opt_2.columns = pd_opt_1.iloc[0], pd_opt_2.iloc[0]
    pd_opt_1.drop('Production forecast for 1 cycle',inplace=True)
    pd_opt_2.drop('Production forecast for 1 cycle',inplace=True)
    pd_opt_1, pd_opt_2 = pd_opt_1[rel_cols_opt], pd_opt_2[rel_cols_opt]
    pd_opt_1['Feed-conversion_ratio'] = pd_opt_1['Total biomass, kg']/pd_opt_1['Monthly feed consumption, kg']
    pd_opt_2['Feed-conversion_ratio'] = pd_opt_2['Total biomass, kg']/pd_opt_2['Monthly feed consumption, kg']
    avg_fcr_1, avg_fcr_2 = pd_opt_1['Feed-conversion_ratio'].mean(), pd_opt_2['Feed-conversion_ratio'].mean()
    avg_grth_rt_1,avg_grth_rt_2 = pd_opt_1['% Growth/day'].mean(), pd_opt_2['% Growth/day'].mean()
    return pd_w3_nrm,pd_w4_nrm,pd_opt_1,pd_opt_2,(avg_fcr_1,avg_fcr_2),(avg_grth_rt_1,avg_grth_rt_2)

pd_w3_nrm,pd_w4_nrm,pd_opt_1,pd_opt_2,(avg_fcr_1,avg_fcr_2),(avg_grth_rt_1,avg_grth_rt_2)=pd_pre_proc()
pd_w3_w4_nrm = pd.concat([pd_w3_nrm,pd_w4_nrm])
np_w3_w4_nrm = pd_w3_w4_nrm.to_numpy()
avg_fcr_1_l = np.asarray([avg_fcr_1]*pd_w3_w4_nrm.shape[0])
avg_grth_1_l = np.asarray([avg_grth_rt_1]*pd_w3_w4_nrm.shape[0])
opt_fcr_grwth = np.stack((avg_fcr_1_l,avg_grth_1_l),axis=-1)

def sigmoid(x): ##sigmoid for both 'input to h_l_1' and 'h_l_1 to h_l_2'...
    return float(1)/(1 + math.exp(-x))


class FishDataset(Dataset):
    def __init__(self,input_params,output_params):
        self.input_params=input_params
        self.output_params=output_params
    def __len__(self):
        assert(len(self.input_params)==len(self.output_params))
        return len(self.input_params)
    def __getitem__(self,index):
        inp = self.input_params[index]
        outp = self.output_params[index]
        return inp, outp

class NNFish(nn.Module):
    def __init__(self,input_dim,hidden_dim_1,hidden_dim_2,output_dim):
        super(NNFish, self).__init__()
        self.input_dim=input_dim
        self.hidden_dim_1=hidden_dim_1
        self.hidden_dim_2=hidden_dim_2
        self.output_dim=output_dim
        self.W1=nn.Linear(input_dim,hidden_dim_1,bias=True)
        self.W2=nn.Linear(hidden_dim_1,hidden_dim_2,bias=True)
        self.W3=nn.Linear(hidden_dim_2,output_dim,bias=True)
    def forward(self,x): #x is num_data_pts x input_dim
        ## print(x, x.shape)
        x=self.W1(x)
        x=torch.sigmoid(x)
        x=self.W2(x)
        x=torch.sigmoid(x)
        x=self.W3(x)
        return x


    
## train the ML model w/ specified weights...
def trainer():
    device = torch.device("cpu")
    model = NNFish(INPUT_DIM,HIDDEN_DIM_1,HIDDEN_DIM_2,OUTPUT_DIM).to(device)
    dataset = DataLoader(dataset=FishDataset(torch.from_numpy(np_w3_w4_nrm.astype(np.float64)),torch.from_numpy(opt_fcr_grwth.astype(np.float64))),shuffle=True)
    optimizer = optim.SGD(model.parameters(),lr=L_RATE,momentum=MOMENTUM)
    for epoch in range(NUM_EPOCH):
        model.train()
        for batch_index,(inp,outp) in enumerate(dataset):
            ##print(inp,outp)
            inp, outp = inp.type('torch.FloatTensor').to(device),outp.type('torch.FloatTensor').to(device)
            outp_pred = model(inp)
            ##print(outp,outp_pred)
            optimizer.zero_grad()
            loss = F.mse_loss(outp_pred,outp)
            loss.backward()
            optimizer.step()
            print("epoch: {}; batch_index: {}; loss: {}".format(epoch+1,batch_index,loss))
    torch.save(model.state_dict(),model_weights)

trainer()

