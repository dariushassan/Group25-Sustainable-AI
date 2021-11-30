# Remember to uncomment lines 36, 37, and 38 based on input dimensions
# Remember to comment the residual stack code at lines 95, 96, and 97 based on input dimensions

# Import required modules
import gc
import tarfile
import numpy as np
from numpy import linalg as la
import h5py
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, LSTM, Input, AlphaDropout, Activation, Reshape, Input
from tensorflow.keras import layers
import tensorflow.keras.models as Model
from tensorflow.keras.regularizers import *
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import GlorotUniform, HeNormal
import tensorflow as tf
import matplotlib.pyplot as plt
import time
from sklearn.metrics import confusion_matrix
import matplotlib
matplotlib.use('Agg')
import pickle
import sys
import operator
from numpy import linalg as la
from math import ceil
from fractions import Fraction


#####################################################################
# L2-PCA
#####################################################################
# Import data

#dim = 128 # uncomment this when the input dimensions are 1/2
#dim = 64 # uncomment this when the input dimensions are 1/4
#dim = 32 # uncomment this when the input dimensions are 1/8

with open('./data/l2/X_train_l2_' + str(dim) + '.pkl', 'rb') as f:
    X_train = pickle.load(f)
X_train = np.stack((X_train[:, :len(X_train[0])//2], X_train[:, len(X_train[0])//2:]), axis=1)
with open('./data/l2/X_test_l2_' + str(dim) + '.pkl', 'rb') as f:
    X_test = pickle.load(f)
X_test = np.stack((X_test[:, :len(X_test[0])//2], X_test[:, len(X_test[0])//2:]), axis=1)
with open('./data/l2/Y_train.pkl', 'rb') as f:
    Y_train = pickle.load(f)
with open('./data/l2/Y_test.pkl', 'rb') as f:
    Y_test = pickle.load(f)
with open('./data/l2/test_idx.pkl', 'rb') as f:
    test_idx = pickle.load(f)
with open('./data/l2/train_idx.pkl', 'rb') as f:
    train_idx = pickle.load(f)
with open('./data/l2/mods.pkl', 'rb') as f:
    mods = pickle.load(f)
with open('./data/l2/snrs.pkl', 'rb') as f:
    snrs = pickle.load(f)
with open('./data/l2/lbl.pkl', 'rb') as f:
    lbl = pickle.load(f)
#####################################################################

    
print('training started')
in_shp = list(X_train.shape[1:])
print(X_train.shape, in_shp)
classes = mods


# Resnet Architecture
def residual_stack(x):
  def residual_unit(y,_strides=1):
    shortcut_unit=y
    # 1x1 conv linear
    y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='relu')(y)
    y = layers.BatchNormalization()(y)
    y = layers.Conv1D(32, kernel_size=5,data_format='channels_first',strides=_strides,padding='same',activation='linear')(y)
    y = layers.BatchNormalization()(y)
    # add batch normalization
    y = layers.add([shortcut_unit,y])
    return y
  
  x = layers.Conv1D(32, data_format='channels_first',kernel_size=1, padding='same',activation='linear')(x)
  x = layers.BatchNormalization()(x)
  x = residual_unit(x)
  x = residual_unit(x)
  # maxpool for down sampling
  x = layers.MaxPooling1D(data_format='channels_first')(x)
  return x


inputs=layers.Input(shape=in_shp)
x = residual_stack(inputs)  # output shape (32,64)
x = residual_stack(x)    # out shape (32,32)
x = residual_stack(x)    # out shape (32,16)    # Comment this when the input dimensions are 1/32 or lower
x = residual_stack(x)    # out shape (32,8)     # Comment this when the input dimensions are 1/16 or lower
x = residual_stack(x)    # out shape (32,4)     # Comment this when the input dimensions are 1/8 or lower
x = Flatten()(x)
x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense1")(x)
x = AlphaDropout(0.1)(x)
x = Dense(128,kernel_initializer="he_normal", activation="selu", name="dense2")(x)
x = AlphaDropout(0.1)(x)
x = Dense(len(classes),kernel_initializer="he_normal", activation="softmax", name="dense3")(x)
x_out = Reshape([len(classes)])(x)
model = tf.keras.models.Model(inputs=inputs, outputs=x_out)
model.compile(loss='categorical_crossentropy', optimizer='adam')
model.summary()
# Set up some params 
nb_epoch = 500     # number of epochs to train on
batch_size = 1024  # training batch size

# Train the Model
# perform training ...
#   - call the main training loop in keras for our network+dataset
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(lr=0.001),
              metrics=['accuracy'])

filepath = './models/l2/resnet_10b_' + str(dim) + 'l2_wts.h5'

start = time.time()
history = model.fit(X_train,
    Y_train,
    batch_size=batch_size,
    epochs=nb_epoch,
    verbose=2,
    validation_split=0.25,
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=0, mode='auto')
    ])
end = time.time()
print(end - start)

# we re-load the best weights once training is finished
model.load_weights(filepath)
model.save('./models/l2/resnet_10b_' + str(dim) + 'l2.h5')

# Evaluate and Plot Model Performance
# Show simple version of performance
score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
print(score)

# Show loss curves 
plt.figure()
plt.title('Training performance')
plt.plot(history.epoch, history.history['loss'], label='train loss+error')
plt.plot(history.epoch, history.history['val_loss'], label='val_error')
plt.legend()
plt.savefig('./results/l2/Train_perf_' + str(dim) + 'l2.png', dpi=100)	#save image


def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues, labels=[]):
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels, rotation=45)
    plt.yticks(tick_marks, labels)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig('./results/l2/conf_mat_' + str(dim) + 'l2.png', dpi=100, bbox_inches='tight')	#save image

# Plot confusion matrix
test_Y_hat = model.predict(X_test, batch_size=batch_size)
conf = np.zeros([len(classes),len(classes)])
confnorm = np.zeros([len(classes),len(classes)])
for i in range(0,X_test.shape[0]):
    j = list(Y_test[i,:]).index(1)
    k = int(np.argmax(test_Y_hat[i,:]))
    conf[j,k] = conf[j,k] + 1
for i in range(0,len(classes)):
    confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
plot_confusion_matrix(confnorm, labels=classes)

# Plot confusion matrix
acc = {}
for snr in snrs:
    # extract classes @ SNR
    test_SNRs = list(map(lambda x: lbl[x][1], test_idx))
    test_X_i = X_test[np.where(np.array(test_SNRs)==snr)]
    test_Y_i = Y_test[np.where(np.array(test_SNRs)==snr)]    

    # estimate classes
    test_Y_i_hat = model.predict(test_X_i)
    conf = np.zeros([len(classes),len(classes)])
    confnorm = np.zeros([len(classes),len(classes)])
    for i in range(0,test_X_i.shape[0]):
        j = list(test_Y_i[i,:]).index(1)
        k = int(np.argmax(test_Y_i_hat[i,:]))
        conf[j,k] = conf[j,k] + 1
    for i in range(0,len(classes)):
        confnorm[i,:] = conf[i,:] / np.sum(conf[i,:])
    #plt.figure()
    #plot_confusion_matrix(confnorm, labels=classes, title="ConvNet Confusion Matrix (SNR=%d)"%(snr))
    cor = np.sum(np.diag(conf))
    ncor = np.sum(conf) - cor
    print("Overall Accuracy for SNR = " + str(snr) + ": ", cor / (cor+ncor))
    acc[snr] = 1.0*cor/(cor+ncor)

# Save results to a pickle file for plotting later
print(acc)
with open('./results/l2/results_resnet_10b_' + str(dim) + 'l2.pkl','wb') as fd:
    pickle.dump( acc , fd )

# Plot accuracy curve
plt.figure()
plt.plot(snrs, list(map(lambda x: acc[x], snrs)))
plt.xlabel("Signal to Noise Ratio")
plt.ylabel("Classification Accuracy")
plt.title("ResNet Classification Accuracy - L2 PCA " + str(Fraction(dim/256)))
plt.savefig('./results/l2/Acc_curve_' + str(dim) + 'l2.png', dpi=100)	#save image
