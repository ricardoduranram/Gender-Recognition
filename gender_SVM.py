import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import face
import util

(x_train,y_train,x_test,y_test) = np.load("./files/face_features2.npy")

image_set = np.load("./files/faces(1,-1).npy").item()
images_test = image_set["images_test"]
labels_test = image_set["labels_test"]

kernel = "linear"
display = False


#Normalize data 
x_train = util.normalize(x_train,axis = 1,kernel = 'ratio')
x_test = util.normalize(x_test, axis = 1,kernel= 'ratio')

x_train = x_train[:,1:]
x_test = x_test[:,1:]


#import ipdb;ipdb.set_trace()
"""
#uncomment this to remove randomness
#np.random.seed(1)

#Add randomness to the Training and Testing
Shuffle = np.random.permutation(len(x_train))
x_train = x_train[Shuffle,:]
y_train = y_train[Shuffle]

Shuffle2 = np.random.permutation(len(x_test))
x_test = x_test[Shuffle2,:]
y_test = y_test[Shuffle2]
"""

#Split Training set  into training and validation sets
x_training = x_train[:400,:]
y_training = y_train[:400]
x_validation = x_train[400:,:]
y_validation = y_train[400:]

#choose the hyper-parameters automatically from a range of values
#Then apply the best parameters to test
c_parameters = []
c_range =  np.arange(1,200,1)
svm_c_error = []
for c_value in c_range:
    model = svm.SVC(kernel=kernel, C=c_value,gamma = 1)
    model.fit(X=x_training, y=y_training)
    error = 1.0 - model.score(x_validation, y_validation)
    svm_c_error.append(error)
    
    
plt.plot(c_range, svm_c_error)
plt.title('{} SVM'.format(kernel))
plt.xlabel('c values')
plt.ylabel('error')
plt.show()        
#plt.xticks(c_range)

#Obtain the the c value that gave the least error
index = np.argmin(svm_c_error)
best_c = c_range[index]


#Test the model with the best hyper-parameters on the validation set
model = svm.SVC(kernel=kernel, C=best_c)
model.fit(X=x_train, y=y_train)
accuracy = model.score(x_validation,y_validation)

#Predict using the Test set
y_pred = model.predict(x_validation)
conf = confusion_matrix(y_validation,y_pred, labels = [-1,1])
target_names = ["female","male"]
report =classification_report(y_validation, y_pred, target_names=target_names)
print("Best C constant: {}".format(best_c))
print("Best Kernel: {}".format(kernel))
print("Accuracy: {}".format(accuracy))
print("confusion matrix: \n{}".format(conf))
print("Classification report:\n {}".format(report) )

#TEst the model with the best hyper-parameters on the test set
model2 = svm.SVC(kernel=kernel, C=best_c)
model2.fit(X=x_train, y=y_train)
accuracy = model2.score(x_test,y_test)

#Predict using the Test set
y_pred = model2.predict(x_test)
conf = confusion_matrix(y_test,y_pred, labels = [-1,1])
report =classification_report(y_test, y_pred, target_names=target_names)

print("Accuracy: {}".format(accuracy))
print("confusion matrix: \n{}".format(conf))
print("Classification report:\n {}".format(report) )

if(display):
    #Retrieve random mis-classified images
    correct_pred = (y_pred == labels_test)
    count = 0
    while (True):
        x = np.random.randint(0,len(correct_pred))
        if(not(correct_pred[x])):
            count+=1
            print("Predicted to: ",y_pred[x]," - Correct value: ",labels_test[x])
            plt.imshow(images_test[x,:])
            plt.show()
        if (count == 10):
            break
