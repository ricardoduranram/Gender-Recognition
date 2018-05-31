import numpy as np
from scipy.spatial import distance
import cv2
import matplotlib.pyplot as plt
import dlib
from imutils import face_utils


#Grouped coordinates based on face structure 
JAWLINE_POINTS = list(range(0, 17))  
RIGHT_EYEBROW_POINTS = list(range(17, 22))  
LEFT_EYEBROW_POINTS = list(range(22, 27))  
NOSE_POINTS = list(range(27, 36))  
RIGHT_EYE_POINTS = list(range(36, 42))  
LEFT_EYE_POINTS = list(range(42, 48))  
MOUTH_OUTLINE_POINTS = list(range(48, 61))  
MOUTH_INNER_POINTS = list(range(61, 68))

#Helper method that helps visualize the vectors extracted from the landmarks.
#The parameters is a collection of tuples in which each contains origin and end 
#coordinates of the vector. 
def draw(image,coordinates):
    for (point1, point2) in coordinates:
        x1 = point1[0]
        y1 = point1[1]
        x2 = point2[0]
        y2 = point2[1]
        cv2.line(image, (x1,y1),(x2,y2), (0, 255, 0), 2)

#Extract potential features by taking the distance between vectors
#Currently only works with the provided 68 landmarks xml file    
def extract_features(dataset,image = None, imdisplay = False):
    features = []
    for (i,shape) in enumerate(dataset):   #landmarks for one face at a time
        jaw = shape[JAWLINE_POINTS]
        eyebrowR = shape[RIGHT_EYEBROW_POINTS]
        eyebrowL = shape[LEFT_EYEBROW_POINTS]
        nose = shape[NOSE_POINTS]
        eyeR = shape[RIGHT_EYE_POINTS]
        eyeL = shape[LEFT_EYE_POINTS]
        mouthOuter = shape[MOUTH_OUTLINE_POINTS]
        mouthInner = shape[MOUTH_INNER_POINTS]
        
        #Potential features on the face
        points =[
            (nose[0],jaw[8]), #Base feature 
            (eyeR[3],eyebrowR[4]),#Upper Right Part of the face
            (eyeR[3],eyebrowL[0]),
            (eyeR[3],nose[6]),    
            (eyeR[3],jaw[7]),
            (eyeR[0],eyebrowR[1]),
            (eyeR[0],eyebrowR[0]),
            (eyeR[0],eyebrowR[4]),
            (eyeR[0],jaw[2]),    
            (eyebrowR[4],eyebrowR[1]),
            (eyebrowR[0], eyebrowR[1]),
            (nose[0],nose[4]),
            (mouthInner[0],nose[5]), #Lower Right Part for the face
            (mouthInner[0],mouthOuter[2]),
            (mouthInner[0],mouthOuter[1]),
            (mouthInner[0],mouthOuter[0]),
            (mouthInner[6],mouthOuter[10]),
            (mouthInner[6],mouthOuter[11]),
            (mouthOuter[10],mouthOuter[11]),
            (mouthOuter[9],jaw[7]),                #*
            (jaw[7],jaw[2]),
            (jaw[7],jaw[4]),
            (jaw[2],jaw[4]),
            (eyeR[3],eyeL[0]),#Middle of the face   #*
            (nose[4],nose[8]),                       
            (nose[6],mouthOuter[3]),                
            (mouthInner[1],mouthOuter[3]),
            (mouthInner[5],mouthOuter[9]),
            (mouthOuter[8],mouthOuter[10]),
            (mouthOuter[9],jaw[8]),
            (jaw[7],jaw[9]),
            (eyeL[0],eyebrowR[4]),#Upper Left Part of the face
            (eyeL[0],eyebrowL[0]),
            (eyeL[0],nose[6]),    
            (eyeL[0],jaw[9]),
            (eyeL[3],eyebrowL[3]),
            (eyeL[3],eyebrowL[4]),
            (eyeL[3],eyebrowL[0]),
            (eyeL[3],jaw[14]),    
            (eyebrowL[0],eyebrowL[3]),
            (eyebrowL[4], eyebrowL[3]),
            (nose[0],nose[8]),
            (mouthInner[2],nose[7]), #Lower Left Part for the face
            (mouthInner[2],mouthOuter[4]),
            (mouthInner[2],mouthOuter[5]),
            (mouthInner[2],mouthOuter[6]),
            (mouthInner[4],mouthOuter[8]),
            (mouthInner[4],mouthOuter[7]),
            (mouthOuter[7],mouthOuter[8]),
            (mouthOuter[9],jaw[9]),
            (jaw[9],jaw[14]),
            (jaw[9],jaw[12]),
            (jaw[14],jaw[12]),
            ]
        #for testing purposes, display the image with the vectors drawn on it
        if (imdisplay):
            draw(image,points)
            cv2.imshow("features",image)
            if (cv2.waitKey(0) == 27):
                cv2.destroyAllWindows()
            
            
        features.append(extract_distance(points))
    features = np.concatenate(features)
    return features

#Computes the magnitude of a vector given initial 
#and end point coordinates.    
def extract_distance(coordinates):
    dist = []
    for (point1, point2) in coordinates:
        dist.append(distance.euclidean(point1,point2))
    dist = np.mat(dist)
    return dist
    
#
def extract_landmarks(data):
    landmarks = []
    i = 0
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("./shape_predictor_68_face_landmarks.dat")
    for (i,image) in enumerate(data):
        rects = detector(image, 1)
        for (j, rect) in enumerate(rects):
            print("Extracting landmarks for image #{}".format(i))
            shape = predictor(image, rect)
            shape = face_utils.shape_to_np(shape)
            reshaped = np.reshape(shape,(1,shape.shape[0],shape.shape[1]))
            landmarks.append(reshaped)
    landmarks = np.concatenate(landmarks)
    return landmarks

#Prints x randonmly choosen images
#This is a helper function in order to visualize the images and 
#their labels.
def visualize(images, label, iter = 5):
    for ran in range(iter):
        index = np.random.randint(0,len(images))
        print("Image #{}    Label: {}".format(ran+1, label[index]))
        plt.imshow(images[index,:])
        plt.show()