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
from scipy.interpolate import interp1d


 
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
n_train_data1 = scipy.delete(n_train_data1,243,1)

#n_train_data1 = scipy.delete(n_train_data1,242,1)
#n_train_data1 = scipy.delete(n_train_data1,312,1)
#n_train_data1 = scipy.delete(n_train_data1,135,1)
                                                       #n_train_data1 has 6000 samples and 1408 features 
train_data = n_train_data1[0:4000 :,]                  #  training data with 4000 pts 
test_data = n_train_data1[4000:6000 :,]                # test_data has 2000 pts  



dev_data = np.genfromtxt("test_data.txt")
n_dev_data = preprocessing.normalize(dev_data)  
n_dev_data = scipy.delete(n_dev_data,227,1)
dev_data = scipy.delete(n_dev_data,243,1)  
#n_dev_data = scipy.delete(n_dev_data,242,1)  
#n_dev_data = scipy.delete(n_dev_data,312,1)
#dev_data = scipy.delete(n_dev_data,135,1)     
                                                            #  development data has 2280 pts 

data_train = train_data
data_test = dev_data
target_train = train_labels
target_test = dev_labels    






#assigning the SVM model 
clf = SVC(C = 0.01 ,gamma = 50)
 
#compute the rms error
def compute_error(x, y, model):
    yfit = model.predict(x)
    return np.sqrt(np.mean((y - yfit) ** 2))
    #print sum(y - ones(len(y))) 




def drawLearningCurve(model):
    sizes = np.linspace(29,4000,10).astype(int)   
    train_error = np.zeros(sizes.shape)
    crossval_error = np.zeros(sizes.shape)
     
    for i,size in enumerate(sizes):
         
        #getting the predicted results of the GaussianNB
        model.fit(data_train[:size,:],target_train[:size])
        predicted = model.predict(data_train)
         
        #compute the validation error
        crossval_error[i] = compute_error(data_test,target_test,model)
         
        #compute the training error
        train_error[i] = compute_error(data_train[:size,:],target_train[:size],model)
        
    #draw the plot

    #train_error[:0] = [0] 
    #sizenew = np.linspace(2,4000,80).astype(int) 
    #f = interp1d(sizes, train_error, kind='quadratic')
    #np.savetxt('trainplt.txt', f)
    #np.savetxt('trainplt1.txt', train_error)
    #print f 
    
    #add = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35]
    #train_error1 = add + train_error 
    
    #sizenew = np.linspace(29,4000,10).astype(int) 
    plt.figure()
    plt.title("learning curve")
    plt.ylim(0,1)
    plt.xlabel("Training examples")
    plt.ylabel("error")
    #plt.plot(sizes,crossval_error,'o-', color="r",lw = 2, label='cross validation error')
    plt.plot(sizes,train_error,'-', color="g", lw = 2, label='training error')
    #plt.plot(sizenew,f(sizenew),'-',color="b",lw=2,label='training error')
    #plt.set_xlabel('cross val error')
    #plt.set_ylabel('rms error')
        
    plt.legend(loc = "best")
    #plt.set_xlim(0,99)
    #plt.set_title('Learning Curve' )
    return  plt
         
drawLearningCurve(clf)

plt.show()
