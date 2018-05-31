import numpy as np
import matplotlib.pyplot as plt
import face
import cv2

dataset = np.load("faces(1,-1).npy").item()
train_samples = dataset["images_train"]
train_labels = dataset["labels_train"]
test_samples = dataset["images_test"]
test_labels = dataset["labels_test"]


landmarks1 = face.extract_landmarks(train_samples)
features1 = face.extract_features(landmarks1)


landmarks2 = face.extract_landmarks(test_samples)
features2 = face.extract_features(landmarks2)

dataset =(features1,train_labels,features2,test_labels)
np.save("face_features2",dataset)



