import numpy as np
import torch
import time
from torch.autograd import Variable
import math

# A simple radio interferometric calibration in PyTorch
# Useful for comparing various optimizers (used in deep learing) in calibration


# use same random initialization for comparisons
torch.manual_seed(69)

# stations
N=62
# baselines
B=N*(N-1)/2
# timeslots (each timeslot is a minibatch)
T=10 # T is the full batch size
nepochs=4 # how many epochs
# weather to use robust (Student's T) noise model instead of Gaussian (L2) noise model
robust_noise=True
robust_nu=2.0 # nu in Student's T noise model

# Jones matrices being estimated, created from leaf variable x
x=torch.rand(8*N,requires_grad=True,dtype=torch.float64)
Jr=x[0:4*N].view(2*N,2)
Ji=x[4*N:8*N].view(2*N,2)

# Model (ground truth)
Jmr=torch.DoubleTensor(2*N,2)
Jmi=torch.DoubleTensor(2*N,2)
Jmr.random_(0,10)
Jmi.random_(0,10)

# visibilities
Vr=torch.DoubleTensor(int(2*B*T),2)
Vi=torch.DoubleTensor(int(2*B*T),2)

# source coherency, at phase center, varying with time T
C=torch.DoubleTensor(int(T*2),2).zero_()
for ci in range(0,T):
 C[2*ci:2*(ci+1),:]=torch.eye(2)*math.cos(float(ci)*math.pi*0.1/T)


# function to find product C=AxB
def mult_AxB(Ar,Ai,Br,Bi):
  return (torch.mm(Ar,Br)-torch.mm(Ai,Bi),torch.mm(Ai,Br)+torch.mm(Ar,Bi))

# function to find product C=AxB^H
def mult_AxBH(Ar,Ai,Br,Bi):
  return (torch.mm(Ar,Br.t())+torch.mm(Ai,Bi.t()),torch.mm(Ai,Br.t())-torch.mm(Ar,Bi.t()))



# produce model visibilities for all baselines, timeslots
for nt in range(0,T):
 Ct=C[2*nt:2*(nt+1),:]
 Zero=torch.DoubleTensor(2,2).zero_()
 boff=0
 for ci in range(0,N):
   Jpr=Jmr[2*ci:2*(ci+1),:]
   Jpi=Jmi[2*ci:2*(ci+1),:]
   for cj in range(ci+1,N):
    Jqr=Jmr[2*cj:2*(cj+1),:]
    Jqi=Jmi[2*cj:2*(cj+1),:]
    (Pr,Pi)=mult_AxBH(Ct,Zero,Jqr,Jqi)
    (V01r,V01i)=mult_AxB(Jpr,Jpi,Pr,Pi)
    Vr[int(2*B*nt)+2*boff:int(2*B*nt)+2*(boff+1),:]=V01r
    Vi[int(2*B*nt)+2*boff:int(2*B*nt)+2*(boff+1),:]=V01i
    boff=boff+1
 

# add noise 
Nr=torch.randn(Vr.shape,dtype=torch.float64)
Ni=torch.randn(Vr.shape,dtype=torch.float64)
Nr=Nr/Nr.norm()
Ni=Ni/Ni.norm()

# this is the simulated data
Vr=Vr+Nr*0.1*Vr.norm()
Vi=Vi+Ni*0.1*Vi.norm()


# model evaluation function  - returns L2 loss of residual
def model_predict(tslot):
 # extract correct offset from data based in tslot=0,...,T-1
 Zero=Variable(torch.DoubleTensor(2,2).zero_())
 boff=0
 rnorm=torch.DoubleTensor(1).zero_()
 inorm=torch.DoubleTensor(1).zero_()
 for ci in range(0,N):
   for cj in range(ci+1,N):
    (Pr,Pi)=mult_AxBH(C[2*tslot:2*(tslot+1),:],Zero,Jr[2*cj:2*(cj+1),:],Ji[2*cj:2*(cj+1),:])
    (V01r,V01i)=mult_AxB(Jr[2*ci:2*(ci+1),:],Ji[2*ci:2*(ci+1),:],Pr,Pi)
    if robust_noise:
     rnorm=rnorm+torch.log(1.0+((Vr[int(2*B*tslot)+2*boff:int(2*B*tslot)+2*(boff+1),:]-V01r).norm()**2)/robust_nu)
    else:
     rnorm=rnorm+(Vr[int(2*B*tslot)+2*boff:int(2*B*tslot)+2*(boff+1),:]-V01r).norm()**2

    if robust_noise:
     inorm=inorm+torch.log(1.0+((Vi[int(2*B*tslot)+2*boff:int(2*B*tslot)+2*(boff+1),:]-V01i).norm()**2)/robust_nu)
    else:
     inorm=inorm+(Vi[int(2*B*tslot)+2*boff:int(2*B*tslot)+2*(boff+1),:]-V01i).norm()**2
    boff=boff+1
 # norm^2 of real+imag
 return rnorm+inorm



# Select the optimizer to use
#optimizer=torch.optim.Adam([x],lr=0.1)
#optimizer=torch.optim.SGD([x],lr=0.001)
from lbfgsnew import LBFGSNew # custom optimizer
optimizer=LBFGSNew([x],history_size=7,max_iter=4,line_search_fn=True,batch_mode=True)


# print initial cost
ll=model_predict(0)
print('time 0.00 epoch 00 tslot 00 loss %e'%ll.item())
start_time=time.time()
for nepoch in range(0,nepochs):
 for nt in range(0,T):
  def closure():
    if torch.is_grad_enabled():
       optimizer.zero_grad()
    loss=model_predict(nt)
    if loss.requires_grad:
      loss.backward()
    return loss

  optimizer.step(closure)
  current_loss=model_predict(nt)
  print('time %f epoch %d tslot %d loss %e'%(time.time()-start_time,nepoch,nt,current_loss.item()))
