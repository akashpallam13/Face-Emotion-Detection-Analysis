                                       FACE EMOTION DETECTION ANALYSIS USING MACHINE             
LEARNING
 By 
 PALLAM AKASH                 21D45A6704
 GUIDE :
 Dr. R Sugumar.
 
                                    ABSTRACT
 • Depression is the most common type of physiological or mood disorder affecting many individuals around the
 globe. Depressed people are more prone to many other problems, like sadness, loneliness, and anxiety.
 • Facial expression recognition is an intelligent human-computer interaction method that has emerged in recent years.
 It has a wide range of applications, such as VR games, medical car e, online education, driving, security, and so on.
 • People’s facial expression is one of the important ways to express their own emotions. Sometimes it is easy to find
 one’s inner thoughts by his expressions.
 • The system is developed the deep learning algorithm such as convolutional neural network. Finally, we can detect
 the emotion like happy, angry, sad, surprise and so on. Then, we can detect the depression from speech. Then,
 finally we can estimate some performance metrics such as accuracy and error rate.
 
                                  INTRODUCTION
 • Even though face and facial feature tracking has been a topic for research for several decades, it is not a solved
 problem. Recently, many promising techniques have been proposed, of which some also show real time performance.
 In this paper, we present a system based on statistical models of face appearance, and show results from a real-time
 tracker.
 • The paper is organized as follows. In Section II the concept of model-based coding is described, followed, in Section
 III, by a discussion on different kind of tracking systems for face animation. In the following sections we describe our
 tracker.
 • InSection IV, the face model, its parameterization, and the model adaptation process (i.e., the tracking) are described.
 
                                  OBJECTIVES
 The main objective of our project is,
To detect or to classify the different types of emotions effectively.
 To implement the deep learning algorithm such as CNN for classification.  To detect the depression from speech effectively.
 To detect the depression based on questions.
 To enhance the overall performance for classification algorithms.
 To implement the framework for user friendly.
 
                               EXISTING SYSTEM
 • In existing system, In this paper our group proposes and designs a lightweight convolutional neural network (CNN)
 for detecting facial emotions in real-time and in bulk to achieve a better classification effect. We verify whether our
 model is effective by creating a real-time vision system.
 • This system employs multi-task cascaded convolutional networks (MTCNN) to complete face detection and
 transmit the obtained face coordinates to the facial emotions classification model we designed firstly. Then it
 accomplishes the task of emotion classification.
 • Not only can our model be stored in an 872.9 kilobytes file, but also its accuracy has reached 67% on the FER-2013
 dataset. And it has good detection and recognition effects on those figures which are out of the dataset.
 
                               DISADVANTAGES
 • The results is low when compared with proposed.
 • Theprediction of emotion is poor
 • The performance is low.
 
                               ADVANTAGES
 • It will recognize easy
 • Theaccuracy is high.
 • The prediction is efficient.
 • Moreuser friendly.
 
                                PROPOSED SYSTEM
 • In proposed system, the depression video dataset was taken as input. Then, we have to implement the
 pre-processing step. In this step, we can convert the videos into frames, then, we have to resize the
 original image. After that, we can extract the features from the pre-processed image by gray level co
occurrence matrix (GLCM).
 • Then, we have to implement the image splitting such as test and train. Test is used for predicting and
 train is used for evaluating the model
 • After that, we have to implement the deep learning algorithm such as Convolutional Neural Network.
 The experimental results shows that some performance metrics such as accuracy and detect the emotion
 or expression. Then, we can upload the speech based on algorithm, it will predict the depression or not.
 • Then, we can answer some set of questions and it will predict the person is depressed or not.
 
                                FLOW DIAGRAM
![image](https://github.com/user-attachments/assets/7159c948-be17-4df8-938f-ced2331ed578)


                               LITERATURE SURVEY
![image](https://github.com/user-attachments/assets/f369b99e-2298-479e-a41b-7288695ec290)
![image](https://github.com/user-attachments/assets/0c275513-2e54-4bbe-ad72-759d687048d3)
![image](https://github.com/user-attachments/assets/57a34762-c468-4ce6-8500-a77b7f1ffdad)
![image](https://github.com/user-attachments/assets/8affab21-0cb3-46e5-86f5-ac02fe3a998b)
![image](https://github.com/user-attachments/assets/b8d66663-9299-41e3-b31e-ef3631c60082)


                                   MODULES
 • Input Video
 • Preprocessing
 • Feature extraction
 • Image splitting 
• Classification
 • Prediction
 • Performance metrics.
 
                                  MODULES DESCRIPTION
    Input Video
 • The dataset contains the images in the form of ‘.mp4’.
 • In this step, we have to read or load the input image by using the imread () function.
 • The input video is used to recognize the face expression.
 • In our process, we are used the tkinter file dialogue box for selecting the input image.
   Preprocessing
 • In our process, we have to resize the image and convert the image into gray scale.
 • To resize an image, you call the resize () method on it, passing in a two-integer tuple argument 
   representing the width and height of the resized image. 
• The function doesn't modify the used image; it instead returns another Image with the new dimensions.
 • Convert an Image to Grayscale in Python Using the Conversion Formula and the matplotlib Library. 
• We can also convert an image to grayscale using the standard RGB to grayscale conversion formula that 
is imgGray = 0.2989 * R + 0.5870 * G + 0.1140 * B.
   Feature extraction
 • The Gray. Level Coocurrence Matrix (GLCM) method is a way of extracting second order 
statistical texture features. The approach has been used in a number of applications, Third and higher 
order textures consider the relationships among three or more pixels.
 • The GLCM functions characterize the texture of an image by calculating how often pairs of pixel 
with specific values and in a specified spatial relationship occur in an image, creating a GLCM, and 
then extracting statistical measures from this matrix.
    Image splitting 
• During the machine learning process, data are needed so that learning can take place. 
• In addition to the data required for training, test data are needed to evaluate the performance of the 
algorithm in order to see how well it works. 
• In our process, we considered 70% of the input dataset to be the training data and the remaining 30% to 
be the testing data.
 • Data splitting is the act of partitioning available data into two portions, usually for cross-validator 
purposes.  
    Classification
 • In our process, we have to implement transfer learning such as CNN.
 • In this step, we can extract the features from the preprocessed image.
 • CNN is a convolutional neural network that is 19 layers deep. You can load a pretrained version of the 
network trained on more than a million images from the ImageNet database. 
• The pretrained network can classify images into 1000 object categories, such as keyboard, mouse, 
pencil, and many animals.
 • Then, we can predict the depression from speech as well as question based.
Performance metrics
 • The Final Result will get generated based on the overall classification and prediction. The 
performance of this proposed approach is evaluated using some measures like,
     Accuracy
 • Accuracy of classifier refers to the ability of classifier. It predicts the class label correctly and the 
accuracy of the predictor refers to how well a given predictor can guess the value of predicted 
attribute for a new data.
 AC= (TP+TN)/ (TP+TN+FP+FN).
 
                             SYSTEM REQUIREMENTS
 SOFTWARE REQUIREMENTS:
 • O/S                    
• Language  
• Front End          
:  Windows 7.
 :  Python
 : Anaconda Navigator – Spyder
 HARDWARE  REQUIREMENTS:
 • System 
• Hard Disk 
• Mouse 
• Keyboard 
• Ram  
:   Pentium IV 2.4 GHz 
:   200 GB
 :   Logitech.
 :   110 keys enhanced
 :      
4GB.

                                      Conclusion
                                      
 • We conclude that, the depression video dataset was taken from dataset repository. We are converted the 
videos into frames. We are extracted the features from pre-processed image by GLCM. 
• We are developed the Deep Learning algorithm such as CNN. Finally, the experimental results shows that 
accuracy. 
• Then, we are predicted or classified the faces to find and recognize the different face emotions or 
depression.
                                   Future Enhancement
                                   
 ➢In Future, To apply the Databases with ground-truth labels, preferably both action units and emotion
specified expressions, are needed for the next generation of systems, which are intended for naturally 
occurring behavior (spontaneous and multimodal) in reallife settings. Work in spontaneous facial 
expression analysis is just now emerging and potentially will have significant impact across a range of 
theoretical and applied topics.

                                     Screenshots
                                     
![image](https://github.com/user-attachments/assets/85e1eb54-0d01-4eee-8407-a23dafb7c790)
![image](https://github.com/user-attachments/assets/be3599eb-83aa-441a-b44b-721ba3fcccf7)
![image](https://github.com/user-attachments/assets/b178e6bb-6f15-4e65-8722-738847f4afb8)
![image](https://github.com/user-attachments/assets/fc1504c3-5fb6-41ac-9f55-590cb46d4f94)
![image](https://github.com/user-attachments/assets/97cf2d41-023a-461f-a1d2-bdacd98c7b72)'
![image](https://github.com/user-attachments/assets/6d5bb07e-55ad-4e56-b6da-50d7fd63b8a4)

                                    REFERENCES
                                    
 1. Ahlberg, J., Forchheimer, R.: Face tracking for model-based coding and face animation. Int.J. Imaging 
Syst. Technol. 13(1), 8–22 (2003)
 2. Baker, S., Kanade, T.: Limits on super-resolution and how to break them. IEEE Trans. PatternAnal. Mach. 
Intell. 24(9), 1167–1183 (2002)
 3. Bartlett, M., Hager, J., Ekman, P., Sejnowski, T.: Measuring facial expressions by computer image 
analysis. Psychophysiology 36, 253–264 (1999)
 4. Bartlett, M., Braathen, B., Littlewort-Ford, G., Hershey, J., Fasel, I., Marks, T., Smith, E., Sejnowski, T., 
Movellan, J.R.: Automatic analysis of spontaneous facial behavior: A final project report. Technical Report 
INC-MPLab-TR-2001.08, Machine Perception Lab, Institute for Neural Computation, University of 
California, San Diego (2001)
 5. Bartlett, M., Littlewort, G., Frank, M., Lainscsek, C., Fasel, I., Movellan, J.: Automatic recognition of facial 
actions in spontaneous expressions. J. Multimed. 1(6), 22–35 (2006)
 6. Black, M.: Robust incremental optical flow. PhD thesis, Yale University (1992)
 7. Black, M., Yacoob, Y.: Tracking and recognizing rigid and non-rigid facial motions using local parametric 
models of image motion. In: Proc. of International Conference on Computer Vision, pp. 374–381 (1995)
 8. Black, M., Yacoob, Y.: Recognizing facial expressions in image sequences using local parameterized 
models of image motion. Int. J. Comput. Vis. 25(1), 23–48 (1997)
 9. Brown, L., Tian, Y.-L.: Comparative study of coarse head pose estimation. In: IEEE Workshop on Motion 
and Video Computing, Orlando (2002)
Thank You
