#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
 Stat 201C
 Author: Zheqi Wu
 Date : 06/10/2018
 Description: boosting GMM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import random
from scipy import linalg
import itertools
from sklearn.mixture import GaussianMixture
import matplotlib as mpl
from scipy.stats import multivariate_normal
import pandas as pd
from scipy import stats
from sklearn.cluster import KMeans


def plot_results(X, Y_, means, covariances, index, title):
    splot = plt.subplot(2, 1, 1 + index)
    for i, (mean, covar, color) in enumerate(zip(
            means, covariances, color_iter)):
        v, w = linalg.eigh(covar)
        v = 2. * np.sqrt(2.) * np.sqrt(v)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        plt.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180. * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
        ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    plt.xlim(-0.5, 3.)
    plt.ylim(-0.5, 2.)
    plt.xticks(())
    plt.yticks(())
    plt.title(title)
    
random.seed(111)
color_iter = itertools.cycle(['gold','navy','darkorange', 'c', 'cornflowerblue'])
"""

"""

def prob_fun(g_mean,g_cov,X_train,k):
    P=np.reshape(np.zeros(X_train.shape[0]),(X_train.shape[0],1))
    for i in range(0,g_mean.shape[0]):
        p=multivariate_normal.pdf(X_train, mean=g_mean[i],cov=g_cov[i])
        P=np.c_[P,p] 
        
    return (P[:,1:(k+1)])

def min_eta(G0,g,k):
    eta_=np.arange(0.01,1.01,0.01)
    i=0
    nelog_l=[]
    #object_f=np.zeros((eta_.shape[0],1))
    #g=prob_fun(gmm.means_, gmm.covariances_,gmm.predict(X_train),k)
    for e in eta_:
        l=np.sum(-np.log(np.multiply(e,g)+np.multiply((1-e),G0)))
        nelog_l.append(l)
        i=i+1

    t=np.where(nelog_l==min(nelog_l))
    eta_best=eta_[t[0]]
    return eta_best
        
def boot_weight(W,X_train):
    W_=[]
    
    for i in range(0,len(W)):
        a=W[i]
        if a>1e-10:
            W_.append(a)
        else:
            W_.append(0)
    W_=W_/np.sum(W_)
    X_index=np.random.choice(X_train.shape[0], X_train.shape[0], p=W_)
    X=X_train[X_index]    
    
    return X,X_index

def tran_g(g,X_,X_train):
    g1=np.reshape(np.zeros(X_train.shape[0]*k),(X_train.shape[0],k))
    for i in range(0,X_train.shape[0]):
        if np.isin(X_[i],X_train):
            index=np.where(X_==X_train[i])[0][0]
            g1[index]=g[i]
    
    return g1

#k=10
def BoostGMM(X_train,Y_train,k):
    p=X_train.shape[1]
    #G0=np.reshape(np.zeros(X_train.shape[0]*k),(X_train.shape[0],k))
    W=np.ones(X_train.shape[0])
    gmm=GaussianMixture(n_components=k, covariance_type='full',random_state=1234).fit(X_train)
    G0=gmm.predict_proba(X_train)
    #G0=prob_fun(gmm.means_, gmm.covariances_,X_train,k)
    
    plot_results(X_train, gmm.predict(X_train), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
    W=W/sum(W)
    item=0
    
    while(item<100):
        X_,X_index=boot_weight(W,X_train)
        gmm=GaussianMixture(n_components=k, covariance_type='spherical',random_state=1234).fit(X_)
        #plot_results(X_, gmm.predict(X_), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
    
        #g1=prob_fun(gmm.means_, gmm.covariances_,X_,k)
        g1=gmm.predict_proba(X_)
        #g=g1[X_index]
        
        g=np.reshape(np.zeros(X_train.shape[0]*k),(X_train.shape[0],k))
        i=0
        count=np.zeros(X_train.shape[0])
        
        for i in range(0,len(X_index)):
            index=X_index[i]
            count[index]+=1
            g[index]=g[index]+g1[i]
            i+=1
        for i in range(0,g.shape[0]):
            if g[i][0]!=0:
                g[i]=g[i]*1/count[i]
        
        eta_=min_eta(G0,g,k)
        print(eta_)
        G1=np.multiply(eta_,g)+np.multiply((1-eta_),G0)
        LG1=np.sum(-np.log(np.multiply(eta_,g)+np.multiply((1-eta_),G0)))
        LG0=np.sum(-np.log(G0+0.000001))
        print(LG1)
        print(LG0)
        #if LG1<LG0:
         #   break
        #Z=np.sum(1/np.sum(G1,axis=1))
        W=1/(1/np.sum(G1,axis=1))
        W=W/np.sum(W)
        ##update
        G0=G1
        print(G0)
        LG0=LG1
        print(G1)
        item+=1
    
    return G0

'''
## simulation
n=200
k=3
mu = [[0,0.5], [1,1], [2,0.5]]
sigma = [[0.1,0.1], [0.25,0.25], [0.15,0.15]]
X1=[]
X2=[]
for i in range(0,k):
    for j in range(0,n):
     x1 = np.random.normal(mu[i][0], sigma[i][0])
     x2 = np.random.normal(mu[i][1], sigma[i][1])
     X1.append(x1)
     X2.append(x2)
label=np.r_[np.ones(n)*0,np.ones(n)*1,np.ones(n)*2]
X = np.c_[X1,X2,label]

X_=X[np.random.choice(X.shape[0],X.shape[0])]
X_train=X_[0:X_.shape[0],0:2]
Y_train=X_[0:X_.shape[0],2]
#X_train, X_test= train_test_split(X_, test_size=0.2)



g=BoostGMM(X_train,Y_train,k=3)
#g=G0
label_result=[]
for i in range(0,X_train.shape[0]):
    label=np.argmax(g[i])
    label_result.append(label)


#label_result=gmm.predict(X_train)
result=np.c_[label_result,Y_train]
df = pd.DataFrame(data=result)
df=df.sort_values(by=[0,1], ascending=True, na_position='first').values
accu=0
for i in range(0,k):
    C=df[df[:,0]==i,0:2]
    print(stats.mode(C[:,1])[0])
    accu=accu+np.sum(C[:,1]==stats.mode(C[:,1])[0])
    
accu=float(accu)/len(label_result)
'''



'''
color = np.array(['gold','navy','darkorange', 'c', 'cornflowerblue'])
label_result=np.array(label_result)
for i in range(0,3):
    plt.scatter(X_train[label_result == i,0], X_train[label_result==i,1], .8, color=color[i])
'''