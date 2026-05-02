""" importing dependencies"""
#import os

# from skimage.data import imread
#import cv2
#import matplotlib.pyplot as plt  # to plot any graph
#import numpy as np
#from keras.layers import Dense, Dropout
#from keras.models import Sequential
#from keras.optimizers import gradient_descent_v2
#from skimage.feature import local_binary_pattern
#from sklearn import preprocessing

# reading haar cascade file from directory to detect fac
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.image as mpimg       # reading images to numpy arrays
import matplotlib.pyplot as plt        # to plot any graph
import matplotlib.patches as mpatches  # to draw a circle at the mean contour
# from skimage.data import imread
from sklearn.ensemble import RandomForestClassifier
import time
import cv2
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import gradient_descent_v2
import pandas as pd
import numpy as np
from sklearn import preprocessing
import os
from skimage.feature import local_binary_pattern

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')



""" face extaction function """

def face_extractor(student_img):
  gray = cv2.cvtColor(student_img, cv2.COLOR_BGR2GRAY)
  faces = face_classifier.detectMultiScale(gray, 1.3, 5)
  if faces is ():
    return None
  for (x, y, w, h) in faces:
    extracted_face = student_img[y:y + h, x:x + w]
  return extracted_face



# creating train variables features and labels
X_train = []   #train features
y_train = []    # train labels

"""Training phase"""

directory = "Images/training"
for root, subdirectories, files in os.walk(directory):
      for file in files:
        imgs=(os.path.join(root,file))
        print(imgs)
        img_aux = cv2.imread(imgs)
        student_face=face_extractor(img_aux)
        # Convert to grayscale as LBP works on grayscale image
        im_gray = cv2.cvtColor(student_face, cv2.COLOR_BGR2GRAY)
        im_gray = cv2.resize(im_gray,(256,256))
        radius = 3
        # Number of points to be considered as neighbourers
        no_points = 8 * radius
        # Uniform LBP is used
        lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
        # Calculate the histogram
        x=np.array(np.unique(lbp, return_counts=True)).T
        # Normalize the histogram
        hist = x[:, 1] / sum(x[:, 1])
        # Append image path in X_name
        # Append histogram to X_name
        X_train.append(hist)
        # # Append class label in y_test
        i=file.split('_')
        y_train.append(i[0])



# creating testing variables
X_test = []  # test features
X_test_name = []  # test labels

""" Testing phase"""

directory = "Images/testing"
for root, subdirectories, files in os.walk(directory):
  for file in files:
      imgs=(os.path.join(root,file))
      print(imgs)

      img_aux = cv2.imread(imgs)
      img_aux = face_extractor(img_aux)
      # Convert to grayscale as LBP works on grayscale image
      im_gray = cv2.cvtColor(img_aux, cv2.COLOR_BGR2GRAY)
      im_gray = cv2.resize(im_gray,(256,256))
      radius = 3
      # Number of points to be considered as neighbourers
      no_points = 8 * radius
      # Uniform LBP is used
      lbp = local_binary_pattern(im_gray, no_points, radius, method='uniform')
      # Calculate the histogram
      x=np.array(np.unique(lbp, return_counts=True)).T
      # Normalize the histogram
      hist = x[:, 1] / sum(x[:, 1])
      # Append image path in X_name
      # Append histogram to X_name
      X_test.append(hist)
      # # Append class label in y_test
      i=file.split('_')
      X_test_name.append(i[0])
      fig = plt.figure(figsize=(20, 8))
      ax = fig.add_subplot(1, 3, 1)
      ax.imshow(im_gray, cmap='gray')
      ax.set_title("gray scale image")
      ax = fig.add_subplot(1, 3, 2)
      ax.imshow(lbp, cmap="gray")
      ax.set_title("LBP converted image")
      ax = fig.add_subplot(1, 3, 3)
      vec = lbp.flatten()
      # vec=np.expand_dims(vec,0)
      ax.hist(vec, bins=2 ** 8)
      ax.set_title("Histogram")
      plt.show()

#Standardization of Train and test values

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = (sc_X.fit_transform((X_train)))
X_test = (sc_X.transform((X_test)))
le = preprocessing.LabelEncoder()
y_train=le.fit_transform(y_train)

""""Model building -MLP(Multi Layer Perceptron) classifier"""
categories = os.listdir("Images/training")
model = Sequential()
#
# # Dense(64) is a fully-connected layer with 64 hidden units.
# # in the first layer, you must specify the expected input data shape:
#
model.add(Dense(64, activation='relu', input_dim=26))
# model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(len(categories), activation='softmax'))

sgd = gradient_descent_v2.SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)

"""Model compiling"""
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

"""Model training"""
history = model.fit(X_train, y_train,
          epochs=10,
          batch_size=5)

# Model saving.....
model.save('lbp_model.h5')

# Prediction results
prediction = model.predict(X_test)
print('prediction',prediction)
index=(np.argmax(prediction))
print(index,'index')
result =categories[index]
print(result)

print(history.history.keys())

# Visualizing losses and accuracy

train_loss = history.history['loss']
train_acc = history.history['accuracy']
no_epochs = range(10)
plt.figure()
# plt.title('epochs vs accuracy')
# plt.plot(no_epochs,train_loss)
# plt.title('epochs vs oss')
# plt.show()
fig, axs = plt.subplots(1,2)
axs[0].plot(no_epochs, train_acc)
axs[0].set_title('epochs vs acc')
axs[1].plot(no_epochs, train_loss)
axs[1].set_title('epochs vs loss')
plt.show()