# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsClassifier

np.set_printoptions(precision=7, threshold=20000,suppress=True)

##################################################################
#hyper-parameters

#number of nearest neighbors 
n_neigh = 10

#number of examples in dataset per class
#for CIFAR-10, there are 10 classes, with exponential data imbalance
num_ex_ds = np.array([5000, 2997, 1796, 1077, 645, 387, 232, 139, 83, 50])

#number of classes in dataset
num_classes = 10

#############################################################
#data

#training feature embeddings file path
train_data=[".../CE_cif_trn.csv"]

#file path to save EOS samples and original FE from the train set
fsav=[".../CE_cif_trn_EOS1.csv"]


#############################################################

for m in range(len(train_data)):
    f = train_data[m]
    
    pdf = pd.read_csv(f)
    x = pdf.to_numpy()
    x.shape 

    #labels
    labs = x[:,0]
    #feature embeddings
    feats = x[:,4:]

    #number of neighbors
    num_nbs = n_neigh
    nn = NearestNeighbors(n_neighbors=num_nbs+1)
    nn.fit(feats)
    dist2, ind2 = nn.kneighbors(feats)

    nds = num_ex_ds 
    
    #number to sample per class to achieve balance
    max_samp = np.max(nds)
    max_samp
    
    nsamp1 = max_samp - nds 

    ndstotal = np.sum(nds)
    
    #accumulate and save base examples, probabilities for sampling,
    #neighbors and labels
    bases = []
    probs = []
    nbs =[]
    labels = []

    
    for d in range(num_classes):

        if nds[d] == max_samp:
            continue
    
        #base class labels
        cx = x[labs==d]
        
        #base class FE
        cf = cx[:,4:]
        dist1, ind1 = nn.kneighbors(cf)
        
        base = []
        prob = []
        nb = []
        lab = []
        for n in range(len(cx)):
            y = labs[ind1[n][1:]]
            
            z = sum(y != d)
            
            #if base example has nearest enemies
            if z > 0:
                ybinary = np.where(y==d,0,1)
                
                ybsum = np.sum(ybinary)
                p = ybinary / ybsum
                prob.append(p)
                
                base.append(ind1[n][0])
                nb.append(ind1[n][1:])
                
                lab.append(d)
            
        bases.extend(base)
        probs.extend(prob)
        nbs.extend(nb)
        labels.extend(lab)
    
    #convert to numpy arrays
    basesn = np.array(bases)
    probsn = np.array(probs)
    nbsn = np.array(nbs)
    labelsn = np.array(labels)
    
    #sampled features and labels
    samples = []
    ysamp = []

    for i in range(num_classes):
        
        if nds[i] == max_samp:
            continue
    
        lab1 = labelsn[labelsn==i]
        base1 = basesn[labelsn==i]
        prob1 = probsn[labelsn==i]
        nb1 = nbsn[labelsn==i]
    
        nsamp = nsamp1[i]
        
        bind = np.random.choice(
                list(range(len(lab1))), size=int(nsamp))
        
        base_indices = base1[bind]
    
        pb = prob1[bind]
    
        neighbor_indices = np.empty([nsamp],dtype=int)
    
        for n in range(nsamp):
            ni = np.random.choice(
                list(range(0, num_nbs)), 1,p=pb[n]) #10
            
            neighbor_indices[n]=ni
    
        X_base = feats[base_indices]
       
        X_neighbor = feats[ind2[base_indices, neighbor_indices]]
    
        diff = X_neighbor - X_base
        
        r = np.random.rand(int(nsamp), 1) 
        
        samps = X_base + np.multiply(r, diff)
        
        samples.extend(samps)
        ylab = np.ones(nsamp) * i
        ysamp.extend(ylab)

    s1 = np.array(samples)
    y1 = np.array(ysamp)

    pd_labs = pd.DataFrame(data=y1,columns=['actual'])
    pd_feats = pd.DataFrame(data=s1) 

    pd_samp = pd.concat([pd_labs, pd_feats],axis=1)
    print('sampled data ',pd_samp.shape)
    
    trn_labs = pd.DataFrame(data=labs,columns=['actual'])
    pd_trn = pd.DataFrame(data=feats)
    pdcombo = pd.concat([trn_labs, pd_trn],axis=1)
    
    print('pdcombo ',pdcombo.shape)
    
    combined = pd.concat([pdcombo, pd_samp],axis=0)
    print()
    print('EOS combined file shape ',combined.shape)
    print()
    
    combined.to_csv(fsav[m],index=False)

















