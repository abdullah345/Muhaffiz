import os
import glob 
import sklearn
from sklearn import cross_validation, grid_search
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import SVC
from sklearn.externals import joblib
import cv2
import numpy as np

#def train_svm_classifer(features, labels, model_output_path):
def somefunction():
    features=[] 
    labels=[] 
    for img in glob.glob("HogDS\LefRes"+'/*.*'):
            
            var_img = cv2.imread(img,0)
            dy=var_img.reshape(-1)
            features.append(dy)
            labels.append(1)
    for img in glob.glob("HogDS\RightwaliRes"+'/*.*'):
            
            var_img = cv2.imread(img,0)
            dy=var_img.reshape(-1)
            features.append(dy)
            labels.append(0)        
            
    features=np.matrix(features)
    #labels=np.matrix(labels)
    
    model_output_path="TrainedModel/"
    
    # save 20% of data for performance evaluation
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(features,labels, test_size=0.2)
    
    param = [
        {
            "kernel": ["linear"],
            "C": [1, 10, 100, 1000]
        },
        {
            "kernel": ["rbf"],
            "C": [1, 10, 100, 1000],
            "gamma": [1e-2, 1e-3, 1e-4, 1e-5]
        }
    ]
    
    # request probability estimation
    svm = SVC(probability=True)
    
    # 10-fold cross validation, use 4 thread as each fold and each parameter set can be train in parallel
    clf = grid_search.GridSearchCV(svm, param,
            cv=10, n_jobs=3, verbose=3)
    
    clf.fit(X_train, y_train)
    
    if os.path.exists(model_output_path):
        joblib.dump(clf.best_estimator_, model_output_path)
    else:
        print("Cannot save trained svm model to {0}.".format(model_output_path))
    
    print("\nBest parameters set:")
    print(clf.best_params_)
    
    y_predict=clf.predict(X_test)
    
    labels=sorted(list(set(labels)))
    print("\nConfusion matrix:")
    print("Labels: {0}\n".format(",".join(labels)))
    print(confusion_matrix(y_test, y_predict, labels=labels))
    
    print("\nClassification report:")
    print(classification_report(y_test, y_predict))
    
if __name__ == '__main__':
    __spec__ = "ModuleSpec(name='builtins', loader=<class '_frozen_importlib.BuiltinImporter'>)"
    somefunction()   