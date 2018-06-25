#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 12 15:30:23 2018

@author: wuzheqi
"""

from sklearn import datasets
import sys
sys.path.append("/Users/cliccuser/Downloads")
from bgmm import *

'''
iris = datasets.load_iris()
X_train = iris.data[:, :4]  # we only take the first two features.
Y_train = iris.target

'''
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_mldata
import matplotlib 
from sklearn.decomposition import PCA


mnist = fetch_mldata('MNIST original')
X = mnist["data"]
y = mnist["target"]

pca = PCA(n_components = 0.7, svd_solver = 'full')
pca.fit(X)
pca_result = pca.transform(X)
X=pca_result

X_train, X_test, Y_train, Y_test = train_test_split(X,y,test_size=0.8)



def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = matplotlib.cm.binary, **options)
    plt.axis("off")

plt.figure(figsize=(9,9))
example_images = np.r_[X[:12000:600], X[13000:30600:600], X[30600:60000:590]]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()


#X_train, X_test, y_train, y_test = train_test_split(X, y)

def accu_(label_result,Y_train,k=3):
    
    result=np.c_[label_result,Y_train]
    df = pd.DataFrame(data=result)
    df=df.sort_values(by=[0,1], ascending=True, na_position='first').values
    accu=0
    for i in range(0,k):
        C=df[df[:,0]==i,0:2]
        print(stats.mode(C[:,1])[0])
        accu=accu+np.sum(C[:,1]==stats.mode(C[:,1])[0])
    
    accu=float(accu)/len(label_result)
    return accu

## B_GMM
g=BoostGMM(X_train,Y_train,k=10)
#g=G0
label_result=[]
for i in range(0,X_train.shape[0]):
    label=np.argmax(g[i])
    label_result.append(label)
accu_BGMM=accu_(label_result,Y_train)

#label_result=gmm.predict(X_train)



## GMM
k=10
gmm=GaussianMixture(n_components=k, covariance_type='full',random_state=1234).fit(X_train)
plot_results(X_train, gmm.predict(X_train), gmm.means_, gmm.covariances_, 0,'Gaussian Mixture')
a=gmm.predict(X_train)
accu_GMM=accu_(gmm.predict(X_train),Y_train)







