from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_digits
import sklearn.cross_validation
from sklearn import preprocessing
from numpy import loadtxt
import scipy
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
import sklearn 

 
#seperating data sets for cross validation
#data_train,data_test,target_train,target_test = cross_validation.train_test_split(digits.data,digits.target,test_size = 0.20, random_state = 42)
 

train_data1 = np.genfromtxt("train_data.txt")
dev_labels = np.genfromtxt("bc3_act_gold_standard.tsv")
dev_labels = dev_labels[:,1]
labels = np.genfromtxt("test_labels.txt")
labels = labels[:,1]
train_labels = labels[0:4000 :,]
test_labels = labels[4000:6000 :,] 
    
 

n_train_data1= preprocessing.normalize(train_data1)
n_train_data1 = scipy.delete(n_train_data1,227,1)
n_train_data1 = scipy.delete(n_train_data1,242,1)
n_train_data1 = scipy.delete(n_train_data1,312,1)
n_train_data1 = scipy.delete(n_train_data1,135,1)
                                                       #n_train_data1 has 6000 samples and 1408 features 
train_data = n_train_data1[0:4000 :,]                  #  training data with 4000 pts 
test_data = n_train_data1[4000:6000 :,]                # test_data has 2000 pts  



dev_data = np.genfromtxt("test_data.txt")
n_dev_data = preprocessing.normalize(dev_data)  
n_dev_data = scipy.delete(n_dev_data,227,1)
n_dev_data = scipy.delete(n_dev_data,242,1)  
n_dev_data = scipy.delete(n_dev_data,312,1)
dev_data = scipy.delete(n_dev_data,135,1)     
                                                            #  development data has 2280 pts 

data_train = train_data
data_test = dev_data
target_train = train_labels
target_test = dev_labels    


clf = SVC(C = 10 , gamma = 20, probability = True )





#print model.get_params
#probs = model.predict_proba(train_data)
#print probs.shape 






def drawLearningCurve(model):
    sizes = np.linspace(29,4000,5).astype(int)   
    train_score = np.zeros(sizes.shape)
    crossval_score = np.zeros(sizes.shape)
    
    for i,size in enumerate(sizes):
        model.fit(data_train[:size,:],target_train[:size])

          
        #compute the validation score
        crossval_score[i] = model.score(data_test,target_test)
         
        #compute the training score
        train_score[i] = model.score(data_train[:size,:],target_train[:size])
    

    print crossval_score
    print train_score
    


    plt.figure()
    plt.title("learning curve")
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    plt.plot(sizes,crossval_score,'o-', color="r",lw = 2, label='cross validation score')
    plt.plot(sizes,train_score,'o-', color="g", lw = 2, label='training score')
    plt.legend(loc = "best")
    return  plt    
    

drawLearningCurve(clf)
plt.show()
