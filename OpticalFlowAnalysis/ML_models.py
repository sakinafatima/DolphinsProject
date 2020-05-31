from collections import Counter

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, recall_score, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import minmax_scale, robust_scale
from sklearn.decomposition import PCA
from sklearn import svm

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=False):
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools

    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.show()
    #reference: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
train = pd.read_csv("TrainingData/TrainVelocity_binary.csv")
print("Shape of Train",train.shape)
train.head()
test = pd.read_csv("TrainingData/validVelocity_binary.csv")
print("Shape of test",test.shape)
test.head()

train=shuffle(train,random_state=49)
train.reset_index(drop=True,inplace=True)
test=shuffle(test,random_state=49)
test.reset_index(drop=True,inplace=True)
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
X_train = pd.DataFrame(train.drop(['label'],axis=1))
Y_train =pd.DataFrame(train.drop(['velocity1'],axis=1))
X_test = pd.DataFrame(test.drop(['label'],axis=1))
Y_test =pd.DataFrame(test.drop(['velocity1'],axis=1))
#modifying scale of data (minMax and robust scale)
feature_list=X_train.columns
x=X_train[feature_list].values
x_scaled=robust_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_train.index)
X_train[feature_list]=df_temp

x=X_test[feature_list].values
x_scaled=robust_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_test.index)
X_test[feature_list]=df_temp
#First Test----------------------Logistic Regression--------------------------without balancing
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1) # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(X_train,Y_train.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])

t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
#2nd Test----------------------Logistic Regression--------------------------with over sampling Smorte
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=49)

Xtrain_smote, ytrain_smote = smote.fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1) # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(Xtrain_smote,ytrain_smote)

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
# Third Test----------------------Logistic Regression--------------------------with balanced class weight without smorte

t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)

t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight="balanced") # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(X_train,Y_train.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#forth test----------------------Logestic Regression--------------------------------with undersampling

from imblearn.under_sampling import TomekLinks
undersample = TomekLinks()

t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)

t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight="balanced") # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(X_train,Y_train.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
