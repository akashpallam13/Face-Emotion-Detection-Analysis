
import numpy as np
import matplotlib.pyplot as plt 

from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import Sequential

import cv2
from skimage.io import imshow

import os
import argparse
import numpy as np
import numpy

import cv2
import warnings
warnings.filterwarnings('ignore')

# Create a VideoCapture object and read from input file
# If the input is the camera, pass 0 instead of the video file name
filename = askopenfilename()

cap = cv2.VideoCapture(filename)
 
 
# Check if camera opened successfully
if (cap.isOpened()== False):
  print("Error opening video stream or file")
else:
      
    Frames = [] 
    # Read until video is completed
    while (cap.isOpened()):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
     
            # Display the resulting frame
            cv2.imshow('Frame', frame)
            Frames.append(frame)
            # Press Q on keyboard to  exit
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
     
        # Break the loop
        else:
            break
     
    # When everything done, release the video capture object
    cap.release()
     
    # Closes all the frames
    cv2.destroyAllWindows()
    warnings.filterwarnings('ignore')


# === GETTING INPUT

from skimage.feature import greycomatrix, greycoprops

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

for iiij in range(0,len(Frames)):
    
    img1 = Frames[iiij]
    print('=====================================')
    print('Frame '+str(iiij))
    print('=====================================')
    plt.imshow(img1)
    plt.title('ORIGINAL FACE IMAGE')
    plt.show()
    
    
    # PRE-PROCESSING
    
    h1=512
    w1=512
    
    dimension = (w1, h1) 
    resized_image1 = cv2.resize(img1,(h1,w1))
    
    # fig = plt.figure()
    # plt.title('RESIZED FACE IMAGE')
    # plt.imshow(resized_image1)
    # plt.show()
    
    
    # ==========================================================================
    
    # FACE Detection
    warnings.filterwarnings('ignore')

    face_cascade = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('opencv-master\data\haarcascades\haarcascade_eye.xml')
    
    #face_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('C:\\opencv\\build\\etc\\haarcascades\\haarcascade_eye.xml')
    
    #face_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
    #eye_cascade = cv2.CascadeClassifier('/usr/local/Cellar/opencv/3.4.0_1/share/OpenCV/haarcascades/haarcascade_eye.xml')
    
    #imgg = cv2.imread('Dataset\1.jpg')
    
    img_face = resized_image1
    grayscale = cv2.cvtColor(img_face, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(grayscale, 1.3, 2)
    
    for (x,y,w,h) in faces:
        
        img_face = cv2.rectangle(img_face,(x,y),(x+w,y+h),(255,0,0),2)
        
        roi_gray = grayscale[y:y+h, x:x+w]
        roi_color = img_face[y:y+h, x:x+w]
        
    
    fig = plt.figure()
    plt.imshow(img_face)
    plt.show()


    
    # -- FEATURE EXTRACTION
    # Face
    
    PATCH_SIZE = 21
    
    image = img_face[:,:,0]
    image = cv2.resize(image,(768,1024))
    warnings.filterwarnings('ignore')

    # select some patches from foreground and background
    
    grass_locations = [(280, 454), (342, 223), (444, 192), (455, 455)]
    grass_patches = []
    for loc in grass_locations:
        grass_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                   loc[1]:loc[1] + PATCH_SIZE])
    
    # select some patches from sky areas of the image
    sky_locations = [(38, 34), (139, 28), (37, 437), (145, 379)]
    sky_patches = []
    warnings.filterwarnings('ignore')

    for loc in sky_locations:
        sky_patches.append(image[loc[0]:loc[0] + PATCH_SIZE,
                                 loc[1]:loc[1] + PATCH_SIZE])
    
    # compute some GLCM properties each patch
    xs = []
    ys = []

    for patch in (grass_patches + sky_patches):
        warnings.filterwarnings('ignore')

        glcm = greycomatrix(patch, distances=[5], angles=[0], levels=256,symmetric=True)


    
    sky_patches0 = np.mean(sky_patches[0])
    sky_patches1 = np.mean(sky_patches[1])
    sky_patches2 = np.mean(sky_patches[2])
    sky_patches3 = np.mean(sky_patches[3])
    
    
    Glcm_fea2 = [sky_patches0,sky_patches1,sky_patches2,sky_patches3]
    
    Test_Features = Glcm_fea2
    warnings.filterwarnings('ignore')

    import pickle
    with open('Train_Features_6.pickle', 'rb') as f:
        Train_features = pickle.load(f)
    
    
    y_trains = np.arange(0,100)
    y_trains[0:50] = 1
    y_trains[50:100] = 20

        
    warnings.filterwarnings('ignore')


    
    from sklearn.ensemble import RandomForestClassifier

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(Train_features, y_trains)
    y_predd = clf.predict([Test_Features])   
    
    
    print('======================')
    print('Classification Results')
    print('======================')
    
    if y_predd == 1:
        print('Depression')
        
    elif y_predd == 2:
        print('Non Depression')
        
    # elif y_predd == 3:
    #     print('Discust')
        
    # elif y_predd == 4:
    #     print('Surprise')
        
    # elif y_predd == 5:
    #     print('Smile')
    print('========================================================')



#=============================== DATA SPLITTING =================================

# === test and train ===
import os 

from sklearn.model_selection import train_test_split

angry_data = os.listdir('1/')

disgust_data = os.listdir('2/')

# fear_data = os.listdir('Input_Data/fear/')

# happy_data = os.listdir('Input_Data/happy/')

# neutral_data = os.listdir('Input_Data/neutral/')

# sad_data = os.listdir('Input_Data/sad/')

# surprise_data = os.listdir('Input_Data/surprise/')


dot1= []
labels1 = []
for img in angry_data:
        # print(img)
        img_1 = cv2.imread('1' + "/" + img)
        img_1 = cv2.resize(img_1,((50, 50)))


        try:            
            gray = cv2.cvtColor(img_1, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_1

        
        dot1.append(np.array(gray))
        labels1.append(0)

        
for img in disgust_data:
    try:
        img_2 = cv2.imread('2'+ "/" + img)
        img_2 = cv2.resize(img_2,((50, 50)))

        

        try:            
            gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
        except:
            gray = img_2
            
        dot1.append(np.array(gray))
        labels1.append(1)
    except:
        None

   for img in fear_data:
     try:
#         img_2 = cv2.imread('Input_Data/fear'+ "/" + img)
#         img_2 = cv2.resize(img_2,((50, 50)))

        

#         try:            
#             gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
#         except:
#             gray = img_2
            
#         dot1.append(np.array(gray))
#         labels1.append(2)
#     except:
#         None
        
        
# for img in happy_data:
#     try:
#         img_2 = cv2.imread('Input_Data/happy/'+ "/" + img)
#         img_2 = cv2.resize(img_2,((50, 50)))

        

#         try:            
#             gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
#         except:
#             gray = img_2
            
#         dot1.append(np.array(gray))
#         labels1.append(3)
#     except:
#         None


        
# for img in sad_data:
#     try:
#         img_2 = cv2.imread('Input_Data/sad/'+ "/" + img)
#         img_2 = cv2.resize(img_2,((50, 50)))

        

#         try:            
#             gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
#         except:
#             gray = img_2
            
#         dot1.append(np.array(gray))
#         labels1.append(4)
#     except:
#         None
        
# for img in surprise_data:
#     try:
#         img_2 = cv2.imread('Input_Data/surprise/'+ "/" + img)
#         img_2 = cv2.resize(img_2,((50, 50)))

        

#         try:            
#             gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
#         except:
#             gray = img_2
            
#         dot1.append(np.array(gray))
#         labels1.append(5)
#     except:
#         None
        
# for img in neutral_data:
#     try:
#         img_2 = cv2.imread('Input_Data/neutral/'+ "/" + img)
#         img_2 = cv2.resize(img_2,((50, 50)))

        

#         try:            
#             gray = cv2.cvtColor(img_2, cv2.COLOR_BGR2GRAY)
            
#         except:
#             gray = img_2
            
#         dot1.append(np.array(gray))
#         labels1.append(6)
#     except:
#         None
        
    
x_train, x_test, y_train, y_test = train_test_split(dot1,labels1,test_size = 0.2, random_state = 101)


#=============================== CLASSIFICATION =================================

#==== VGG19 =====

from tensorflow.keras.models import Sequential

from tensorflow.keras.applications.vgg19 import VGG19
vgg = VGG19(weights="imagenet",include_top = False,input_shape=(50,50,3))

for layer in vgg.layers:
    layer.trainable = False
from tensorflow.keras.layers import Flatten,Dense
model = Sequential()
model.add(vgg)
model.add(Flatten())
model.add(Dense(1,activation="sigmoid"))
model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping
checkpoint = ModelCheckpoint("vgg19.h5",monitor="val_acc",verbose=1,save_best_only=True,
                             save_weights_only=False,period=1)
earlystop = EarlyStopping(monitor="val_acc",patience=5,verbose=1)

from keras.utils import to_categorical


y_train1=np.array(y_train)
y_test1=np.array(y_test)

train_Y_one_hot = to_categorical(y_train1)
test_Y_one_hot = to_categorical(y_test)




x_train2=np.zeros((len(x_train),50,50,3))
for i in range(0,len(x_train)):
        x_train2[i,:,:,:]=x_train2[i]

x_test2=np.zeros((len(x_test),50,50,3))
for i in range(0,len(x_test)):
        x_test2[i,:,:,:]=x_test2[i]



history = model.fit(x_train2,y_train1,batch_size=50,
                    epochs=2,validation_data=(x_train2,y_train1),
                    verbose=1,callbacks=[checkpoint,earlystop])
print("===================================================")
print("---------- Convolutional Neural Network ----------")
print("==================================================")
print()
accuracy=history.history['accuracy']
accuracy=max(accuracy)
accuracy=100-accuracy
print()
print("Accuracy is :",accuracy,'%')























