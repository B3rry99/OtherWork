import array 
import keras 
from keras.models import Sequential,Model 
from tensorflow.python.keras.models import Input 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import LeakyReLU 
from sklearn.model_selection import train_test_split 
from keras.datasets import fashion_mnist 
import numpy as np 
from keras.utils import to_categorical 
import matplotlib.pyplot as plt 
import matplotlib.pyplot as plt 
%matplotlib inline 
from PIL import Image 
import PIL 
import os 
from matplotlib.image import imread 
import matplotlib.pyplot as plt 
import cv2 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\DATA" 
## RESIZING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
for file in f: 
if file.endswith('.png'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
if im.size!=(32, 32): 
im=im.resize((32, 32),Image.LANCZOS) 
im.save(pat.replace(".png",".jpg")) 
os.remove(pat) 
i+=1 
print(i,end='\r') 
elif file.endswith('.jpg'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
if im.size!=(32, 32): 
im=im.resize((32, 32),Image.LANCZOS) 
im.save(pat) 
i+=1 
print(i,end='\r') 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\DATA" 
## GRAYSCALING  TRAINING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
for file in f: 
if file.endswith('.png'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
                 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\TEST" 
 
 
## RESIZING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
    for file in f: 
        if file.endswith('.png'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                if im.size!=(32, 32): 
                    im=im.resize((32, 32),Image.LANCZOS) 
                im.save(pat.replace(".png",".jpg")) 
            os.remove(pat) 
            i+=1 
            print(i,end='\r') 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                if im.size!=(32, 32): 
                    im=im.resize((32, 32),Image.LANCZOS) 
                    im.save(pat) 
                    i+=1 
                    print(i,end='\r') 
 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\TEST" 
 
## GRAYSCALING  TRAINING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
    for file in f: 
        if file.endswith('.png'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
                 
                 
 
def ImageArray(folder): 
    string = str(folder) 
    path = os.path.join("C:\\Users\\Jack\\Desktop\\Uni 
Work\\NeuralNetworks\\DATA",string) 
    #images =[] 
    # r=root, d=directories, f = files 
    for r, d, f in os.walk(path): 
        for file in f: 
            if file.endswith('.jpg'): 
                pat=os.path.join(r, file) 
                with Image.open(pat) as im: 
                    Allimages.append(cv2.imread(pat,0))   #replace "Allimages" with 
"images" to revert to original 
            elif file.endswith('.jpg'): 
                pat=os.path.join(r, file) 
                with Image.open(pat) as im: 
                    Allimages.append(cv2.imread(pat,0))   #replace "Allimages" with 
"images" to revert to original 
 
Allimages =[] 
for i in range(58): 
    ImageArray(i) 
                 
train_X = np.array(Allimages) 
 
Test = [] 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\TEST" 
for r, d, f in os.walk(path): 
    for file in f: 
        if file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                Test.append(cv2.imread(pat,0))   #replace "Allimages" with "images" 
to revert to original 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                Test.append(cv2.imread(pat,0))   #replace "Allimages" with "images" 
to revert to original 
 
test_X = np.array(Test) 
def TrueValues(folder): 
string = str(folder) 
path = os.path.join("C:\\Users\\Jack\\Desktop\\Uni 
Work\\NeuralNetworks\\DATA",string) 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
for file in f: 
if file.endswith('.jpg'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
Values.append(i) 
elif file.endswith('.jpg'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
Values.append(i) 
TrainValues =[] 
for i in range(58): 
Values = [] 
TrueValues(i) 
TrainValues.append(Values) 
train_Y = [item for sublist in TrainValues for item in sublist] 
train_Y = np.array(train_Y) 
test_Y = []                     
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\TEST" 
for i in range(10): 
number = str(i) 
number2 = "00" + number 
for r, d, f in os.walk(path): 
for file in f: 
if file.startswith(number2): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
test_Y.append(i) 
for i in range(10,58): 
number = str(i) 
number2 = "0" + number 
for r, d, f in os.walk(path): 
for file in f: 
if file.startswith(number2): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
test_Y.append(i) 
test_Y = np.array(test_Y) 
classes = np.unique(train_Y) 
nClasses = len(classes) 
print('Total number of outputs : ', nClasses) 
print('Output classes : ', classes) 
plt.figure(figsize=[5,5]) 
# Display the first image in training data 
plt.subplot(121) 
plt.imshow(train_X[0,:,:], cmap='gray') 
plt.title("Ground Truth : {}".format(train_Y[0])) 
# Display the first image in testing data 
plt.subplot(122) 
plt.imshow(test_X[0,:,:], cmap='gray') 
plt.title("Ground Truth : {}".format(test_Y[0])) 
train_X = train_X.reshape(-1, 32,32, 1) 
test_X = test_X.reshape(-1, 32,32, 1) 
train_X.shape, test_X.shape 
train_X = train_X.astype('float32') 
test_X = test_X.astype('float32') 
train_X = train_X / 255. 
test_X = test_X / 255. 
train_X.shape, test_X.shape 
# Change the labels from categorical to one-hot encoding 
train_Y_one_hot = to_categorical(train_Y) 
test_Y_one_hot = to_categorical(test_Y) 
# Display the change for category label using one-hot encoding 
print('Original label:', train_Y[0]) 
print('After conversion to one-hot:', train_Y_one_hot[0]) 
train_X,valid_X,train_label,valid_label = train_test_split(train_X, train_Y_one_hot, 
test_size=0.2, random_state=13) 
train_Y_one_hot 
batch_size = 16 
epochs = 20 
num_classes = 58 
model = Sequential() 
model.add(Conv2D(32, kernel_size=(3, 
3),activation='linear',input_shape=(32,32,1),padding='same')) 
model.add(LeakyReLU(alpha=0.0001)) 
model.add(MaxPooling2D((2, 2),padding='same')) 
model.add(Conv2D(64, (3, 3), activation='linear',padding='same')) 
model.add(LeakyReLU(alpha=0.0001)) 
model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) 
model.add(Conv2D(128, (3, 3), activation='linear',padding='same')) 
model.add(LeakyReLU(alpha=0.0001)) 
model.add(MaxPooling2D(pool_size=(2, 2),padding='same')) 
model.add(Flatten()) 
model.add(Dense(128, activation='linear')) 
model.add(LeakyReLU(alpha=0.0001)) 
model.add(Dense(num_classes, activation='softmax')) 
model.compile(loss=keras.losses.categorical_crossentropy, 
optimizer=keras.optimizers.Adam(),metrics=['accuracy']) 
model.summary() 
train_X.shape 
Model_Train = model.fit(train_X, train_label, 
batch_size=batch_size,epochs=20,verbose=1,validation_data=(valid_X, valid_label)) 
test_eval = model.evaluate(test_X, test_Y_one_hot, verbose=0) 
print('Test loss:', test_eval[0]) 
print('Test accuracy:', test_eval[1]) 
k=list(Model_Train.history.keys())# access to dictionary keys 
a=list(Model_Train.history.values())# access to dictionary keys 
accuracy = a[3] 
val_accuracy = a[1] 
loss = a[2] 
val_loss= a[0] 
epochs = range(len(accuracy)) 
fig1 = plt.gcf() 
plt.plot(Model_Train.history['accuracy']) 
plt.plot(Model_Train.history['val_accuracy']) 
plt.axis(ymin=0.4,ymax=1.1) 
plt.grid() 
plt.title('Training and Validation Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epochs') 
plt.legend(['train', 'validation']) 
plt.show() 
plt.plot(Model_Train.history['loss']) 
plt.plot(Model_Train.history['val_loss']) 
plt.grid() 
plt.title('Training and Validation Loss') 
plt.ylabel('Loss') 
plt.xlabel('Epochs') 
plt.legend(['train', 'validation']) 
plt.show() 
batch_size = 16 
epochs = 20 
num_classes = 58 
model2 = Sequential() 
model2.add(Conv2D(32, kernel_size=(3, 
3),activation='linear',padding='same',input_shape=(32,32,1))) 
model2.add(LeakyReLU(alpha=0.0001)) 
model2.add(MaxPooling2D((2, 2),padding='same')) 
model2.add(Dropout(0.25)) 
model2.add(Conv2D(64, (3, 3), activation='linear',padding='same')) 
model2.add(LeakyReLU(alpha=0.0001)) 
model2.add(MaxPooling2D(pool_size=(2, 2),padding='same')) 
model2.add(Dropout(0.25)) 
model2.add(Conv2D(128, (3, 3), activation='linear',padding='same')) 
model2.add(LeakyReLU(alpha=0.0001)) 
model2.add(MaxPooling2D(pool_size=(2, 2),padding='same')) 
model2.add(Dropout(0.4)) 
model2.add(Flatten()) 
model2.add(Dense(128, activation='linear')) 
model2.add(LeakyReLU(alpha=0.0001)) 
model2.add(Dropout(0.3)) 
model2.add(Dense(num_classes, activation='softmax')) 
model2.summary() 
model2.compile(loss=keras.losses.categorical_crossentropy, 
optimizer=keras.optimizers.Adam(),metrics=['accuracy']) 
#train the model 
fashion_train_dropout = model2.fit(train_X, train_label, 
batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(valid_X, 
valid_label)) 
test_eval = model2.evaluate(test_X, test_Y_one_hot, verbose=1) 
print('Test loss:', test_eval[0]) 
print('Test accuracy:', test_eval[1]) 
kk=list(fashion_train_dropout.history.keys())# access to dictionary keys 
vv=list(fashion_train_dropout.history.values())# access to dictionary keys 
accuracy = vv[3] 
val_accuracy = vv[1] 
loss = vv[2] 
val_loss= vv[0] 
epochs = range(len(accuracy)) 
plt.plot(epochs, accuracy, 'b*', label='Validation accuracy') 
plt.plot(epochs, val_accuracy, 'g', label='Training accuracy') 
plt.title('Training and validation accuracy') 
plt.legend() 
plt.figure() 
plt.plot(epochs, loss, 'b*', label='Validation loss') 
plt.plot(epochs, val_loss, 'g', label='Training loss') 
plt.title('Training and validation loss') 
plt.legend() 
plt.show() 
predicted_classes = model2.predict(test_X) 
predicted_classes = np.argmax(np.round(predicted_classes),axis=1) 
test_accuracy=sum(predicted_classes==test_Y)/len(test_Y) 
test_accuracy 
ResNet50 Code 
import matplotlib.pyplot as plt 
import numpy as np 
import os 
import PIL 
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers 
from tensorflow.python.keras.layers import Dense, Flatten 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.optimizers import Adam 
from PIL import Image 
import shutil 
import array 
import keras 
from keras.models import Sequential,Model 
from tensorflow.python.keras.models import Input 
from keras.layers import Dense, Dropout, Flatten 
from keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import BatchNormalization 
from tensorflow.keras.layers import LeakyReLU 
from sklearn.model_selection import train_test_split 
from keras.utils import to_categorical 
%matplotlib inline 
from matplotlib.image import imread 
import cv2 
import pathlib 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\D1" 
## RESIZING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
for file in f: 
if file.endswith('.png'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
if im.size!=(32, 32): 
im=im.resize((32, 32),Image.LANCZOS) 
im.save(pat.replace(".png",".jpg")) 
os.remove(pat) 
elif file.endswith('.jpg'): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
if im.size!=(32, 32): 
im=im.resize((32, 32),Image.LANCZOS) 
im.save(pat) 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\T1" 
## RESIZING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
    for file in f: 
        if file.endswith('.png'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                if im.size!=(32, 32): 
                    im=im.resize((32, 32),Image.LANCZOS) 
                im.save(pat.replace(".png",".jpg")) 
            os.remove(pat) 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                if im.size!=(32, 32): 
                    im=im.resize((32, 32),Image.LANCZOS) 
                    im.save(pat) 
                     
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\D1" 
 
## GRAYSCALING  TRAINING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
    for file in f: 
        if file.endswith('.png'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
                 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\T1" 
 
## GRAYSCALING  TRAINING IMAGES 
i=0 
# r=root, d=directories, f = files 
for r, d, f in os.walk(path): 
    for file in f: 
        if file.endswith('.png'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
        elif file.endswith('.jpg'): 
            pat=os.path.join(r, file) 
            with Image.open(pat) as im: 
                image_array = np.array(PIL.Image.open(pat)) 
                for i in range(len(image_array)): 
                    for j in range(len(image_array[i])): 
                        blue = image_array[i, j, 0] 
                        green = image_array[i, j, 1] 
                        red = image_array[i, j, 2] 
                        grayscale_value = blue * 0.114 + green * 0.587 + red * 0.299 
                        image_array[i,j] = grayscale_value 
                im = PIL.Image.fromarray(image_array)     
                im.save(pat) 
# TRAINING SET 
data = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\DATA" 
img_height,img_width = 32,32 
batch_size=16 
train = tf.keras.preprocessing.image_dataset_from_directory( 
  data, 
  validation_split=0.2, 
  seed=100, 
  subset="training", 
  label_mode="categorical", 
  image_size=(img_height, img_width), 
  batch_size=batch_size) 
 
# VALIDATION SET 
val = tf.keras.preprocessing.image_dataset_from_directory( 
  data, 
  validation_split=0.2, 
  seed=32,   
  subset="validation", 
  label_mode="categorical", 
  image_size=(img_height, img_width), 
  batch_size=batch_size) 
  
classes = train.class_names 
resnet_model = Sequential() 
pretrainedmodel = tf.keras.applications.ResNet50( 
    include_top=False, 
    input_shape=(32,32,3), 
    weights='imagenet', 
    classes = 58, 
    pooling='avg') 
for layer in pretrainedmodel.layers: 
    layer.trainable=False 
 
resnet_model.add(pretrainedmodel) 
resnet_model.add(Flatten()) 
resnet_model.add(Dense(128, activation='relu')) 
resnet_model.add(Dense(58, activation='softmax')) 
resnet_model.summary() 
resnet_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),loss=kera
 s.losses.categorical_crossentropy,metrics=['accuracy']) 
epochs = 20 
history = resnet_model.fit(train,validation_data=val,epochs=epochs) 
fig1 = plt.gcf() 
plt.plot(history.history['accuracy']) 
plt.plot(history.history['val_accuracy']) 
plt.axis(ymin=0.4,ymax=1.1) 
plt.grid() 
plt.title('Training and Validation Accuracy') 
plt.ylabel('Accuracy') 
plt.xlabel('Epochs') 
plt.legend(['train', 'validation']) 
plt.show() 
plt.plot(history.history['loss']) 
plt.plot(history.history['val_loss']) 
plt.grid() 
plt.title('Training and Validation Loss') 
plt.ylabel('Loss') 
plt.xlabel('Epochs') 
plt.legend(['train', 'validation']) 
plt.show() 
test_Y = [] 
path = "C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks\\TEST" 
for i in range(10): 
number = str(i) 
number2 = "00" + number 
for r, d, f in os.walk(path): 
for file in f: 
if file.startswith(number2): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
test_Y.append(i) 
for i in range(10,58): 
number = str(i) 
number2 = "0" + number 
for r, d, f in os.walk(path): 
for file in f: 
if file.startswith(number2): 
pat=os.path.join(r, file) 
with Image.open(pat) as im: 
test_Y.append(i) 
test_dir = 'C:\\Users\\Jack\\Desktop\\Uni Work\\NeuralNetworks' 
test_dir = pathlib.Path(test_dir) 
testing = list(test_dir.glob('TEST/*')) 
predictions = [] 
for i in range(1994): 
image=cv2.imread(str(testing[i])) 
image_resized= cv2.resize(image, (img_height,img_width)) 
image=np.expand_dims(image_resized,axis=0) 
pred=resnet_model.predict(image) 
output_class = classes[np.argmax(pred)] 
output_class2 = int(output_class) 
predictions.append(output_class2) 
matching = [i for i, j in zip(test_Y, predictions) if i == j]    
test_accuracy = ((len(matching)*100)/1994) 
print("test accuracy is: ", test_accuracy) 