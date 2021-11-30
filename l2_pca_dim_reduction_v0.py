import os, random, sys
import numpy as np
import pickle

print("Loading Dataset")
## Open the dataset and load the data.
with open("./data/RML2016.10b.dat", "rb") as p:
    Xd = pickle.load(p, encoding='latin1')
print("shape of Xd", np.shape(Xd))  # Print Shape of dataset.
print("Loading Dataset Done")
print("type of Xd:", type(Xd))

snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
X = []
lbl = []
for mod in mods:
    for snr in snrs:
        X.append(Xd[(mod,snr)])
        for i in range(Xd[(mod,snr)].shape[0]):  lbl.append((mod,snr))
X = np.vstack(X)
print("shape of X", np.shape(X))

# Partition the dataset into training and testing datasets
np.random.seed(2016)     # Random seed value for the partitioning (Also used for random subsampling)
n_examples = X.shape[0]
n_train = n_examples // 2 # Split dataset in half.
train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False) # Choose randomly train/test samples.
test_idx = list(set(range(0,n_examples))-set(train_idx)) # Assign test and train samples.

X_train = X[train_idx] # Train dataset.
X_test =  X[test_idx]  # Test dataset.

def to_onehot(yy):
    yy1 = np.zeros([len(yy), max(yy)+1])
    yy1[np.arange(len(yy)),yy] = 1
    return yy1

Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

#####################################################################
# L2-PCA
#####################################################################

# PCA Setup.
from sklearn.decomposition import PCA
pca_rank = [4, 8, 16, 32, 64]  # Rank of the output matrix of pca.
#pca1 = PCA(n_components=pca_rank[0]*2) # Initialize the PCA function.
#pca2 = PCA(n_components=pca_rank[1]*2) # Initialize the PCA function.
pca3 = PCA(n_components=pca_rank[2]*2) # Initialize the PCA function.
pca4 = PCA(n_components=pca_rank[3]*2) # Initialize the PCA function.
pca5 = PCA(n_components=pca_rank[4]*2) # Initialize the PCA function.

# Data setup.
X_train = X_train.transpose((1, 0, 2))
X_train = np.append(X_train[0], X_train[1], axis=1)

# Run L2-PCA on the data.
#pca_apply1 = pca1.fit(X_train) # Calculate the subspaces.
#pca_apply2 = pca2.fit(X_train) # Calculate the subspaces.
pca_apply3 = pca3.fit(X_train) # Calculate the subspaces.
pca_apply4 = pca4.fit(X_train) # Calculate the subspaces.
pca_apply5 = pca5.fit(X_train) # Calculate the subspaces.

#X_train_l2_8  = pca_apply1.transform(X_train) # Project on the calculated subspace.
#X_train_l2_16 = pca_apply2.transform(X_train) # Project on the calculated subspace.
X_train_l2_32 = pca_apply3.transform(X_train) # Project on the calculated subspace.
X_train_l2_64 = pca_apply4.transform(X_train) # Project on the calculated subspace.
X_train_l2_128= pca_apply5.transform(X_train) # Project on the calculated subspace.

#print('Shape of X_train after PCA', np.shape(X_train_l2_8))
#print('Shape of X_train after PCA', np.shape(X_train_l2_16))
print('Shape of X_train after PCA', np.shape(X_train_l2_32))
print('Shape of X_train after PCA', np.shape(X_train_l2_64))
print('Shape of X_train after PCA', np.shape(X_train_l2_128))

# Do the same for the test dataset.
X_test = X_test.transpose((1, 0, 2))
X_test = np.append(X_test[0], X_test[1], axis=1)

#X_test_l2_8   = pca_apply1.transform(X_test)
#X_test_l2_16  = pca_apply2.transform(X_test)
X_test_l2_32  = pca_apply3.transform(X_test)
X_test_l2_64  = pca_apply4.transform(X_test)
X_test_l2_128 = pca_apply5.transform(X_test)

#print('Shape of X_test after PCA', np.shape(X_test_l2_8))
#print('Shape of X_test after PCA', np.shape(X_test_l2_16))
print('Shape of X_test after PCA', np.shape(X_test_l2_32))
print('Shape of X_test after PCA', np.shape(X_test_l2_64))
print('Shape of X_test after PCA', np.shape(X_test_l2_128))


##### SAVING THE RESULTS IN A FILE

pickle.dump(test_idx, open("./data/l2/test_idx.pkl","wb"))
pickle.dump(Y_test, open("./data/l2/Y_test.pkl","wb"))
#pickle.dump(X_test_l2_8, open("X_test_l2_8.pkl","wb"))
#pickle.dump(X_test_l2_16, open("X_test_l2_16.pkl","wb"))
pickle.dump(X_test_l2_32, open("./data/l2/X_test_l2_32.pkl","wb"))
pickle.dump(X_test_l2_64, open("./data/l2/X_test_l2_64.pkl","wb"))
pickle.dump(X_test_l2_128, open("./data/l2/X_test_l2_128.pkl","wb"))

pickle.dump(train_idx, open("./data/l2/train_idx.pkl","wb"))
pickle.dump(Y_train, open("./data/l2/Y_train.pkl","wb"))
#pickle.dump(X_train_l2_8, open("X_train_l2_8.pkl","wb"))
#pickle.dump(X_train_l2_16, open("X_train_l2_16.pkl","wb"))
pickle.dump(X_train_l2_32, open("./data/l2/X_train_l2_32.pkl","wb"))
pickle.dump(X_train_l2_64, open("./data/l2/X_train_l2_64.pkl","wb"))
pickle.dump(X_train_l2_128, open("./data/l2/X_train_l2_128.pkl","wb"))

pickle.dump(mods, open("./data/l2/mods.pkl", "wb"))
pickle.dump(snrs, open("./data/l2/snrs.pkl", "wb"))
pickle.dump(lbl, open("./data/l2/lbl.pkl", "wb"))

#### ALL DONE!

