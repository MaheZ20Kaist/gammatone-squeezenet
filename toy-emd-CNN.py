#!/usr/bin/env python
# coding: utf-8

# In[1]:



import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

X = np.load("x_imn_env2.npz")
"""car/abnormal
car/normal
conveyor/abnormal
conveyor/normal
train/abnormal
train/normal"""


# In[2]:


import numpy as np
import gtgram
from scipy.io import wavfile
from pydub import AudioSegment
import matplotlib.pyplot as plt
import matplotlib
from multiprocessing import Pool
import os
import tensorflow as tf
import pandas as pd
import emd
from lime import lime_image
from lime import submodular_pick

from skimage.segmentation import mark_boundaries

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications import VGG16


# In[3]:


import time
import shutil
import numpy as np
import os
import pandas as pd
from multiprocessing import Pool
import emd


# In[4]:


path = 'dataset'

labels = []
signal_path = []

for cls1 in os.listdir(path):
    for cls2 in os.listdir(os.path.join(path, cls1)):
        label = os.path.join(cls1,cls2)
        for signal in os.listdir(os.path.join(path,cls1,cls2)):
            labels.append(label)
            signal_path.append(os.path.join(path,cls1,cls2,signal))


# In[5]:


signal_path[:5]


# In[6]:


labels[:5]


# In[7]:


import gtgram
from scipy.io import wavfile
from pydub import AudioSegment
'''gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=8000)])'''
def get_mono_sig(file_path:str, top_n=6):
    fs, _ = wavfile.read(file_path)
    sig_mono = AudioSegment.from_wav(file_path)
    sig_mono = sig_mono.set_channels(1)
    sig_mono = np.array(sig_mono.get_array_of_samples()).astype(float)
    
    imf = emd.sift.sift(sig_mono)
    
    return imf[:, :top_n].sum(axis = 1), fs

def get_gtgram_texture(resultant_signal, fs):
    res_scale = 1
    windowTime = 0.05
    hop_time = 0.32/res_scale
    channels = 32*res_scale
    
    specg = gtgram.gtgram(wave = resultant_signal, 
                            fs = fs, 
                            window_time = windowTime, 
                            hop_time = hop_time, 
                            channels = channels, 
                            f_min = 30)
    return specg


# In[8]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[9]:


y = le.fit_transform(labels)
le.classes_


# In[10]:


y = y.reshape((*y.shape, 1))


# In[11]:



# X_load = []
# with Pool(24) as p:
#     X_load.append(p.map(get_mono_sig, signal_path))
# X_load = X_load[0]
# X = []
# for item in X_load:
#     tex = get_gtgram_texture(item[0], item[1])
#     tex = np.resize(tex, (32,32))
#     X.append(tex)
# print(np.array(X).shape)
# from numpy import savez_compressed
# savez_compressed('x_imn_env2.npz', np.array(X))


# In[12]:


X = X["arr_0"]
X = X.reshape(*X.shape, 1)


# In[13]:


print("X: ", X.shape)
print("y: ", y.shape)


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4, shuffle=True)


# In[15]:


X_train = X_train/255
X_test = X_test/255


# In[16]:


n_classes = 6
num_epochs = 200
BATCH_SIZE = 16
INPUT_SIZE = (32,32)
TOTAL_TRAIN_SAMPLES = X_train.shape[0]
TOTAL_TEST_SAMPLES = X_test.shape[0]


# In[17]:


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience= 150, restore_best_weights=True
)

bnmomemtum=0.9
def fire(x, squeeze, expand):
    y  = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation='relu', padding='same')(x)
    #y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
    y1 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=1, activation='relu', padding='same')(y)
    #y1 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y1)
    y3 = tf.keras.layers.Conv2D(filters=expand//2, kernel_size=3, activation='relu', padding='same')(y)
    #y3 = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y3)
    return tf.keras.layers.concatenate([y1, y3])

def fire_module(squeeze, expand):
    return lambda x: fire(x, squeeze, expand)

def encoder(input_shape = (INPUT_SIZE[0],INPUT_SIZE[1],1)):
    x = tf.keras.layers.Input(shape=input_shape) 
    #y = tf.keras.layers.Reshape(target_shape = (32,32,1))(x)
    #y = augmentation(x)
    y = tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', use_bias=True, activation='relu')(x)
    #y = tf.keras.layers.BatchNormalization(momentum=bnmomemtum)(y)
    y = fire_module(24, 48)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(48, 96)(y)
    y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    y = fire_module(48, 96)(y)
    #  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #  y = fire_module(64, 128)(y)
    #  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #  y = fire_module(48, 96)(y)
    #  y = tf.keras.layers.MaxPooling2D(pool_size=2)(y)
    #  y = fire_module(24, 48)(y)
    y = tf.keras.layers.Conv2D(kernel_size=3, filters=16, padding='same', activation='relu', name="block14_sepconv2_act")(y)
    y = tf.keras.layers.GlobalAveragePooling2D()(y)
    #  y = tf.keras.layers.Dense(64)(y)
    y, _ = tf.keras.layers.MultiHeadAttention(num_heads=2, key_dim=64, attention_axes=(1))(query=y, key=y, value=y, return_attention_scores=True)
    #  y = tf.keras.layers.Dense(64,activation = 'relu')(y)
    y = tf.keras.layers.Dense(n_classes, activation='softmax')(y)

    model = tf.keras.Model(x, y)
    return model


# In[18]:


# model = encoder()
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(INPUT_SIZE[0], INPUT_SIZE[1], 1)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(6, activation='softmax')
])
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[19]:


model.summary()


# In[20]:


history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = num_epochs, 
           verbose = 1,validation_split = 0.11,callbacks = [early_stopping])


# In[21]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[22]:


y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors
y_labels = np.argmax(y_pred,axis = 1)
# Convert validation observations to one hot vectors
true_labels = y_test
# compute the confusion matrix
print(true_labels.shape, y_labels.shape)
cm = confusion_matrix(true_labels, y_labels)
target_names = ['TCar_A','TCar_N','TConveyor_A','TConveyor_N','TTrain_A','TTrain_N']


# In[23]:


print (classification_report(y_labels,true_labels, target_names = target_names))


# In[24]:


import seaborn as sn
import pandas as pd
c=24
array = cm
fig, ax = plt.subplots(figsize=(10,5))
df_cm = pd.DataFrame(array)
sn.set(font_scale=1) 
sn.heatmap(df_cm,cmap = 'Blues', annot=True, fmt='g',annot_kws={"size": 12}, ax=ax) # font size
plt.show()


# In[25]:


# AUC ROC plots
from sklearn.metrics import roc_curve, auc
n_classes = 6
fpr = dict()
tpr = dict()
roc_auc = dict()
# ypred = finetuning_model.predict(labeled_test_data, batch_size = None, verbose = 2,steps = None, callbacks = None)


from sklearn.preprocessing import LabelBinarizer
from itertools import cycle
lw = 2
lb = LabelBinarizer()
yenc = lb.fit_transform(y_labels)


for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(yenc[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(yenc.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue","red","purple", "cyan"])
for i, color,dt in zip(range(n_classes), colors, target_names):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(dt, roc_auc[i]),
    )

plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Some extension of Receiver operating characteristic to multiclass")
plt.legend(loc="lower right")
plt.show()


# In[26]:


loss = history.history['loss']
accuracy = history.history['accuracy']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = np.arange(1, len(val_accuracy)+1)

plt.plot(epochs, loss, label="training loss")
plt.plot(epochs, val_loss,  label="validation loss")
plt.xlabel("Epochs")
plt.ylabel("Metric")
plt.title("Loss Curves")
plt.legend()
plt.show()


# In[27]:



plt.plot(epochs, accuracy, label="training accuracy")
plt.plot(epochs, val_accuracy, label="validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Metric")
plt.title("Accuracy Curves")
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:




