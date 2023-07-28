#!/usr/bin/env python
# coding: utf-8

# In[87]:


import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)


# In[88]:


X = np.load("x_imn.npz")


# In[89]:


X_inm = X["arr_0"]


# In[90]:


X_inm.shape


# In[91]:


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

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
# # session.close()
# gpus = tf.config.list_physical_devices('GPU')



gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=8000)])

# tf.config.experimental.set_memory_growth(gpus[0], True)
# tf.config.experimental.set_memory_growth(gpus[1], True)

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# if gpus:
#   # Restrict TensorFlow to only allocate 4GB of memory on the first GPU
#   try:
#     tf.config.experimental.set_virtual_device_configuration(
#         gpus[0],
#         [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])
#     logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
#   except RuntimeError as e:
#     # Virtual devices must be set before GPUs have been initialized
#     print(e)


# In[92]:


def get_mono_sig(file_path:str, top_n=6):
    fs, _ = wavfile.read(file_path)
    sig_mono = AudioSegment.from_wav(file_path)
    sig_mono = sig_mono.set_channels(1)
    sig_mono = np.array(sig_mono.get_array_of_samples()).astype(float)
    
    imf = emd.sift.sift(sig_mono)
    
    return imf[:, :top_n].sum(axis = 1), fs

def get_gtgram_texture(resultant_signal):
#     fs, _ = wavfile.read(file_path)
#     sig_mono = AudioSegment.from_wav(file_path)
#     sig_mono = sig_mono.set_channels(1)
#     sig_mono = np.array(sig_mono.get_array_of_samples()).astype(float)
    
#     imf = emd.sift.sift(sig_mono)
    
#     resultant_signal = imf[:, :top_n].sum(axis = 1)
    
    res_scale = 1
    windowTime = 0.05
    hop_time = 0.32/res_scale
    channels = 32*res_scale
    
    specg = gtgram.gtgram(wave = resultant_signal[0], 
                            fs = resultant_signal[1], 
                            window_time = windowTime, 
                            hop_time = hop_time, 
                            channels = channels, 
                            f_min = 30)
    return specg


# In[93]:


path = "dataset"
environment = ['slider', 'fan', 'valve', 'pump']


# In[94]:


image_paths_normal = []
image_paths_abnormal = []

image_normal_labels = []
image_abnormal_labels = []

for item in environment:
    # print("1")
    for file1 in os.listdir(os.path.join(path, item)):
        # print("2")
        for file2 in os.listdir(os.path.join(path,item, file1)):
            # print("3")
            # for file3 in os.listdir(os.path.join(path,item, file1, file2)):
                # print(file3)
                if file2 == "normal":
                    for file4 in os.listdir(os.path.join(path,item, file1, file2)):
                        image_paths_normal.append(os.path.join(path, item, file1, file2, file4))
                        image_normal_labels.append(file2+"/"+item)
                elif file2 == "abnormal":
                    for file4 in os.listdir(os.path.join(path,item, file1, file2)):
                        image_paths_abnormal.append(os.path.join(path, item, file1, file2, file4))
                        image_abnormal_labels.append(file2+"/"+item)


# In[9]:


# X = []
# for item in image_paths_normal:
#     X.append(get_gtgram_texture(item, 7))
# for item in image_paths_abnormal:
#     X.append(get_gtgram_texture(item, 7))


# In[10]:


# X_load = []
# with Pool(24) as p:
#     X_load.append(p.map(get_mono_sig, image_paths_normal))


# In[11]:


# with Pool(24) as p:
#     X_load.append(p.map(get_mono_sig, image_paths_abnormal))


# In[12]:


# X_1 = np.array(X_load[0])
# X_2 = np.array(X_load[1])


# In[13]:


# X_inm = np.append(X_1, X_2, axis=0)


# In[14]:


# from numpy import savez_compressed
# savez_compressed('x_imn_env2.npz', X_inm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[95]:


X_load = []
with Pool(6) as p:
    X_load.append(p.map(get_gtgram_texture, X_inm))


# In[96]:


# with Pool(24) as p:
#     X_load.append(p.map(get_gtgram_texture, X_inm))


# In[97]:


X = np.array(X_load[0])


# In[98]:


X = X.reshape(*X.shape, 1)


# In[99]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


# In[100]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()
labels = np.append(np.array(image_normal_labels), np.array(image_abnormal_labels))
y = le.fit_transform(labels)


# In[101]:


pd.DataFrame(y).value_counts().sort_index().to_list()


# In[102]:


dictionary = pd.DataFrame(labels).value_counts().to_dict()
new_dictionary = {}
for k, v in zip(list(dictionary.keys()), list(dictionary.values())):
    new_dictionary[v] = k[0]
new_dictionary


# In[103]:


target_names = []
for key in pd.DataFrame(y).value_counts().sort_index().to_list():
    target_names.append(new_dictionary[key])


# In[104]:


y = y.reshape((*y.shape, 1))


# In[105]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=4, shuffle=True)


# In[106]:


X_train = X_train/255
X_test = X_test/255


# In[107]:


n_classes = 8
num_epochs = 300
BATCH_SIZE = 32
INPUT_SIZE = (32,32)
TOTAL_TRAIN_SAMPLES = X_train.shape[0]
TOTAL_TEST_SAMPLES = X_test.shape[0]


# In[28]:


# train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# train_ds = train_ds.shuffle(TOTAL_TRAIN_SAMPLES).batch(BATCH_SIZE)

# test_ds = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# test_ds = test_ds.shuffle(TOTAL_TEST_SAMPLES).batch(BATCH_SIZE)


# In[29]:



# augmentation = tf.keras.Sequential([
#     tf.keras.layers.RandomFlip(),
# ])
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


# In[30]:


model = encoder()
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 5e-4),loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


# In[31]:


model.summary()


# In[32]:


history = model.fit(X_train, y_train, batch_size = BATCH_SIZE, epochs = num_epochs, 
           verbose = 1,validation_split = 0.11,callbacks = [early_stopping])


# In[33]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# In[34]:


y_pred = model.predict(X_test)
# Convert predictions classes to one hot vectors
y_labels = np.argmax(y_pred,axis = 1)
# Convert validation observations to one hot vectors
true_labels = y_test
# compute the confusion matrix
print(true_labels.shape, y_labels.shape)
cm = confusion_matrix(true_labels, y_labels)
# target_names = ['TCar_A','TCar_N','TConveyor_A','TConveyor_N','TTrain_A','TTrain_N']


# In[35]:


print (classification_report(y_labels,true_labels, target_names = target_names))


# In[36]:


import seaborn as sn
import pandas as pd
c=24
array = cm
fig, ax = plt.subplots(figsize=(10,5))
df_cm = pd.DataFrame(array)
sn.set(font_scale=1) 
sn.heatmap(df_cm,cmap = 'Blues', annot=True, fmt='g',annot_kws={"size": 12}, ax=ax) # font size
plt.show()


# In[37]:


# AUC ROC plots
from sklearn.metrics import roc_curve, auc
n_classes = 8
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


# In[38]:


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


# In[39]:


plt.plot(epochs, accuracy, label="training accuracy")
plt.plot(epochs, val_accuracy, label="validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Metric")
plt.title("Accuracy Curves")
plt.legend()
plt.show()


# In[40]:


# import shap


# In[ ]:





# In[ ]:


# shap.explainers.deep


# In[41]:


# del X_inm
# del X
# del y


# In[42]:


# Lime
def pred(img):
    return model.predict(img[:, :, :, 0])


# In[43]:


explainer = lime_image.LimeImageExplainer()
idx = 150
img = np.append(np.append(X_train[idx],X_train[idx], axis=2), X_train[idx], axis=2)
exp = explainer.explain_instance(img,
                                 pred,
                                 top_labels=5,
                                 hide_color=0,
                                 num_samples=10000,)

# plt.imshow(exp.segments)
# plt.axis('off')
# plt.show()

# plt.imshow(exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=True)[0])
# plt.axis('off')
# plt.show()


# In[44]:


from skimage.segmentation import mark_boundaries

temp_1, mask_1 = exp.get_image_and_mask(exp.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
temp_2, mask_2 = exp.get_image_and_mask(exp.top_labels[0], positive_only=False, num_features=10, hide_rest=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,15))
ax1.imshow(mark_boundaries(temp_1, mask_1))
ax2.imshow(mark_boundaries(temp_2, mask_2))
ax1.axis('off')
ax2.axis('off')


# In[45]:



##Grad cam


# In[46]:


last_conv_layer = "block14_sepconv2_act"
# model_copy = copy.deepcopy(model)
model.layers[-1].activation = None


# In[47]:


import matplotlib.cm as c_map
from IPython.display import Image, display

def get_heatmap(vectorized_image, model, last_conv_layer, pred_index=None):
    '''
    Function to visualize grad-cam heatmaps
    '''
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    # Gradient Computations
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradient_model(vectorized_image[:, :, :, :1])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # Normalize the heatmap
    return heatmap.numpy()

# plt.matshow(get_heatmap(vectorized_image, model, last_conv_layer))
# plt.show()
def superimpose_gradcam(imgs, heatmap, output_path="grad_cam_image.jpg", alpha=0.4):
    '''
    Superimpose Grad-CAM Heatmap on image
    '''
    # imgs = image.load_img(img_path)
    # img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap) # Back scaling to 0-255 from 0 - 1
    jet = c_map.get_cmap("jet") # Colorizing heatmap
    jet_colors = jet(np.arange(256))[:, :3] # Using RGB values
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = jet_heatmap.resize(INPUT_SIZE)
    jet_heatmap = image.img_to_array(jet_heatmap)
    
    # print(jet_heatmap.shape)
    # print(img.shape)
    
    superimposed_img = jet_heatmap * alpha + imgs # Superimposing the heatmap on original image
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(output_path) # Saving the superimposed image
    display(Image(output_path)) # Displaying Grad-CAM Superimposed Image

from tensorflow.keras.preprocessing import image
def vectorize_image(img, size):
    '''
    Vectorize the given image to get a numpy array
    '''
    img = image.array_to_img(img)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0) # Adding dimension to convert array into a batch of size (1,299,299,3)
    return array
superimpose_gradcam(image.array_to_img(img[:, :, :]), get_heatmap(vectorize_image(img[:, :, :], size=INPUT_SIZE), model, last_conv_layer))
# model.layers[-1].activation = "softmax"


# In[48]:


img.shape


# In[49]:


i = plt.imread("grad_cam_image.jpg")
plt.imshow(i)


# In[ ]:





# In[50]:


#shap

import random
idx = random.randint(0, X.shape[0]-4)

import shap
def model_output(img_):
    # img_ = img_.resize(1, *INPUT_SIZE, 1)
    return model.predict(img_)

mask = shap.maskers.Image("inpaint_telea", (*INPUT_SIZE, 1))
explainer = shap.Explainer(model_output, mask, output_names=target_names )
shap_values = explainer(X[idx:idx+2], max_evals=5000, batch_size=100, outputs=shap.Explanation.argsort.flip[:8])
shap.image_plot(shap_values, labels = np.array([target_names, target_names]))
print(y[idx:idx+2])


# In[ ]:





# In[57]:


#deeplift

import shap
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

e = shap.GradientExplainer(model, background)

# shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
# shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough


shap_values = e.shap_values(X_test[1:5])
shap.image_plot(shap_values, -X_test[1:5])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[67]:


'''
label 0 -> idx 18
label 1 -> idx 7
label 2 -> idx 0
label 3 -> idx 14
label 4 -> idx 2
label 5 -> idx 6
label 6 -> idx 9
label 7 -> idx 1

'''


# In[69]:


l = [18, 7, 0, 14, 2, 6, 9, 1]


# In[71]:


X_test[l].shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#### SHAP


# In[72]:


import shap
background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]

e = shap.GradientExplainer(model, background)

# shap.explainers._deep.deep_tf.op_handlers["AddV2"] = shap.explainers._deep.deep_tf.passthrough
# shap.explainers._deep.deep_tf.op_handlers["Einsum"] = shap.explainers._deep.deep_tf.passthrough


shap_values = e.shap_values(X_test[l])
shap.image_plot(shap_values, -X_test[l])


# In[ ]:





# In[ ]:





# In[ ]:


### GradCAM


# In[73]:


import matplotlib.cm as c_map
from IPython.display import Image, display

def get_heatmap(vectorized_image, model, last_conv_layer, pred_index=None):
    '''
    Function to visualize grad-cam heatmaps
    '''
    gradient_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer).output, model.output]
    )

    # Gradient Computations
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = gradient_model(vectorized_image[:, :, :, :1])
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap) # Normalize the heatmap
    return heatmap.numpy()

# plt.matshow(get_heatmap(vectorized_image, model, last_conv_layer))
# plt.show()
def superimpose_gradcam(imgs, heatmap, output_path="grad_cam_image.jpg", alpha=0.4):
    '''
    Superimpose Grad-CAM Heatmap on image
    '''
    # imgs = image.load_img(img_path)
    # img = image.img_to_array(img)

    heatmap = np.uint8(255 * heatmap) # Back scaling to 0-255 from 0 - 1
    jet = c_map.get_cmap("jet") # Colorizing heatmap
    jet_colors = jet(np.arange(256))[:, :3] # Using RGB values
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = image.array_to_img(jet_heatmap)
    # jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = jet_heatmap.resize(INPUT_SIZE)
    jet_heatmap = image.img_to_array(jet_heatmap)
    
    # print(jet_heatmap.shape)
    # print(img.shape)
    
    superimposed_img = jet_heatmap * alpha + imgs # Superimposing the heatmap on original image
    superimposed_img = image.array_to_img(superimposed_img)

    superimposed_img.save(output_path) # Saving the superimposed image
    display(Image(output_path)) # Displaying Grad-CAM Superimposed Image

from tensorflow.keras.preprocessing import image
def vectorize_image(img, size):
    '''
    Vectorize the given image to get a numpy array
    '''
    img = image.array_to_img(img)
    array = image.img_to_array(img)
    array = np.expand_dims(array, axis=0) # Adding dimension to convert array into a batch of size (1,299,299,3)
    return array

# model.layers[-1].activation = "softmax"


# In[78]:


for idx, arr in enumerate(X_test[l]):
    img = np.append(np.append(arr,arr, axis=2), arr, axis=2)
    superimpose_gradcam(image.array_to_img(img[:, :, :]), get_heatmap(vectorize_image(img[:, :, :], size=INPUT_SIZE), model, last_conv_layer), output_path="class_"+str(idx)+"_gradcam_.jpg")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[82]:


for i, j in zip(np.arange(8), le.inverse_transform(np.arange(8))):
    print(i, ":", j)


# In[ ]:




