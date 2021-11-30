import gc
import tarfile
import numpy as np
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LSTM, Input, AlphaDropout, Activation, Reshape, Input
from tensorflow.keras import layers
import tensorflow.keras.models as Model
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import pickle

'''
# open extracted dataset
f = h5py.File("./2018.01/GOLD_XYZ_OSC.0001_1024.hdf5", "r")
dir_path = "./extractdataset"
modu_snr_size = 1200

# assign data
for modu in range(24):
  X_list = []
  Y_list = []
  Z_list = []
  start_modu = modu * 106496
  for snr in range(26):
    start_snr = start_modu + (snr * 4096)
    idx_list = np.random.choice(range(0, 4096), size=modu_snr_size, replace=False)
    Xd = f['X'][start_snr:start_snr+4096][idx_list]
    X_list.append(Xd)
    Y_list.append(f['Y'][start_snr:start_snr+4096][idx_list])
    Z_list.append(f['Z'][start_snr:start_snr+4096][idx_list])

  filename = dir_path + '/part' + str(modu) + '.h5'
  fw = h5py.File(filename, 'w')
  fw['X'] = np.vstack(X_list)
  fw['Y'] = np.vstack(Y_list)
  fw['Z'] = np.vstack(Z_list)
  fw.close()

# close file
f.close()

'''
'''
print("Loading dataset...")
for i in range(0, 24):
  filename = "./extractdataset/part" + str(i) + ".h5"
  f = h5py.File(filename, "r")
  X = f["X"][:]
  Y = f["Y"][:]
  Z = f["Z"][:]
  f.close()

  # split data into train/test
  n_examples = X.shape[0]
  n_train = int(n_examples * 0.7)
  train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
  test_idx = list(set(range(0,n_examples)) - set(train_idx))
  if i == 0:
    X_train = X[train_idx]
    Y_train = Y[train_idx]
    Z_train = Z[train_idx]
    X_test = X[test_idx]
    Y_test = Y[test_idx]
    Z_test = Z[test_idx]

  # assign split
  X_train = np.vstack((X_train, X[train_idx]))
  Y_train = np.vstack((Y_train, Y[train_idx]))
  Z_train = np.vstack((Z_train, Z[train_idx]))
  X_test = np.vstack((X_test, X[test_idx]))
  Y_test = np.vstack((Y_test, Y[test_idx]))
  Z_test = np.vstack((Z_test, Z[test_idx]))
print("Done loading dataset")
'''

def gendata(fp, nsamples):
    global snrs, mods, train_idx, test_idx, lbl
    with open(fp, 'rb') as p:
        Xd = pickle.load(p, encoding='latin1')
    snrs,mods = map(lambda j: sorted(list(set(map(lambda x: x[j], Xd.keys())))), [1,0])
    X = []  
    lbl = []
    print(mods, snrs)
    for mod in mods:
        for snr in snrs:
            X.append(Xd[(mod,snr)])
            for i in range(Xd[(mod,snr)].shape[0]):
                lbl.append((mod,snr))
    X = np.vstack(X)
    
    print('Length of lbl', len(lbl))
    print('shape of X', X.shape)

    np.random.seed(2016)
    n_examples = X.shape[0]
    n_train = n_examples//2
    train_idx = np.random.choice(range(0,n_examples), size=n_train, replace=False)
    test_idx = list(set(range(0,n_examples))-set(train_idx))
    X_train = X[train_idx]
    X_test =  X[test_idx]
    keys = Xd.keys()
    def to_onehot(yy):
        yy1 = np.zeros([len(yy), max(yy)+1])
        yy1[np.arange(len(yy)),yy] = 1
        return yy1
    Y_train = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), train_idx)))
    Y_test = to_onehot(list(map(lambda x: mods.index(lbl[x][0]), test_idx)))

    return (X_train,X_test,Y_train,Y_test)
 
maxlen = 128

   
X_train, X_test, Y_train, Y_test = gendata("./data/RML2016.10b.dat",maxlen)
print('using version 10b dataset')
print(xtrain1.shape)
test_SNRs = map(lambda x: lbl[x][1], test_idx)
train_SNRs = map(lambda x: lbl[x][1], train_idx)
train_snr = lambda snr: xtrain1[np.where(np.array(train_SNRs)==snr)]
test_snr = lambda snr: ytrain1[np.where(np.array(train_SNRs)==snr)]
classes = mods

print("--"*50)
print("Training data:",X_train.shape)
print("Training labels:",Y_train.shape)
print("Testing data",X_test.shape)
print("Testing labels",Y_test.shape)
print("--"*50)

# residual stack
def res_stack(x, filters, kernel_size, pool_size):
  # 1x1 Conv Linear
  x = Conv2D(filters, (1, 1), padding="same", kernel_initializer="glorot_normal", data_format="channels_first")(x)
  # res unit 1
  x_skip = x
  x = Conv2D(filters, kernel_size, padding="same", activation="relu", kernel_initializer="glorot_uniform", data_format="channels_first")(x)
  x = Conv2D(filters, kernel_size, padding="same", kernel_initializer="glorot_normal", data_format="channels_first")(x)
  x = layers.add([x, x_skip])
  x = Activation("relu")(x)
  # res unit 2
  x_skip = x
  x = Conv2D(filters, kernel_size, padding="same", activation="relu", kernel_initializer="glorot_normal", data_format="channels_first")(x)
  x = layers.add([x, x_skip])
  x = Activation("relu")(x)
  # max pooling
  x = MaxPooling2D(pool_size=pool_size, strides=pool_size, padding="valid", data_format="channels_first")(x)
  return x

# residual network
def ResNet(X_input, in_shp):
  # input
  x = Reshape([1,128,2], input_shape=in_shp)(X_input)
  # res stack
  x = res_stack(x, filters=32, kernel_size=(3,2), pool_size=(2,2))
  x = res_stack(x, filters=32, kernel_size=(3,1), pool_size=(2,1))
  x = res_stack(x, filters=32, kernel_size=(3,1), pool_size=(2,1))
  x = res_stack(x, filters=32, kernel_size=(3,1), pool_size=(2,1))
  x = res_stack(x, filters=32, kernel_size=(3,1), pool_size=(2,1))
  x = res_stack(x, filters=32, kernel_size=(3,1), pool_size=(2,1))
  # fc/selu
  x = Flatten(data_format="channels_first")(x)
  x = Dense(128, activation="selu", kernel_initializer="glorot_normal")(x)
  x = AlphaDropout(0.3)(x)
  # fc/softmax
  x = Dense(len(classes), kernel_initializer="glorot_normal")(x)
  x = Activation("softmax")(x)
  return x

# start training
in_shp = X_train.shape[1:]
X_input = Input(in_shp)
x = ResNet(X_input, in_shp)
model = Model.Model(inputs=X_input, outputs=x)
filepath = "./models/res_72w_wts_16.h5"

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=["accuracy"])
model.summary()

print("Starting training...")
start = time.time()
history = model.fit(X_train,
  Y_train,
  batch_size=1000,
  epochs=100,
  verbose=2,
  validation_data=(X_test, Y_test),
  callbacks = [
    tf.keras.callbacks.ModelCheckpoint(filepath, monitor="val_loss", save_best_only=True, mode="auto"),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, mode="auto")
  ])

end = time.time()
print("Done training")
print(end - start)

# save model
model.save("./models/res_72w_16.h5")
model.load_weights(filepath)

# loss curves
plt.figure()
plt.title("Training performance")
plt.plot(history.epoch, history.history["loss"], label="train loss+error")
plt.plot(history.epoch, history.history["val_loss"], label="val_error")
plt.legend()
plt.savefig("./train_perf_res72w.png", dpi=100)
