import itertools
from collections import Counter

import numpy as np
import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from vecstack import stacking
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, classification_report, recall_score, balanced_accuracy_score, \
    accuracy_score
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.preprocessing import minmax_scale, robust_scale
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

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
    plt.xlabel('Predicted label')
    plt.show()
    #reference: https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
train = pd.read_csv("TrainingData/datawithclusters/train_clusters.csv")
print("Shape of Train",train.shape)
train.head()
test = pd.read_csv("TrainingData/datawithclusters/valid_clusters.csv")
print("Shape of test",test.shape)
test.head()
train=shuffle(train,random_state=49)
train.reset_index(drop=True,inplace=True)
test=shuffle(test,random_state=49)
test.reset_index(drop=True,inplace=True)
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
train_outcome = pd.crosstab(index=train["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print(train_outcome)
test_outcome = pd.crosstab(index=test["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print(test_outcome)
train=shuffle(train,random_state=49)
train.reset_index(drop=True,inplace=True)
test=shuffle(test,random_state=49)
test.reset_index(drop=True,inplace=True)
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
X_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label',"velocity_meters", "kmeans","path"],axis=1))
Y_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity_pixels", "velocity_meters","kmeans", "hdbscan","path"],axis=1))
X_test = pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label',"velocity_meters", "kmeans","path"],axis=1))
Y_test =pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity_pixels", "velocity_meters","kmeans", "hdbscan","path"],axis=1))
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features in X_train:",num_cols.size)
num_cols = Y_train._get_numeric_data().columns
print("Number of numeric features in Y_train:",num_cols.size)

num_cols = X_test._get_numeric_data().columns
print("Number of numeric features in X_test:",num_cols.size)
num_cols = Y_test._get_numeric_data().columns
print("Number of numeric features in Y_test:",num_cols.size)

names_of_predictors = list(X_train.columns.values)
print("names of predictors X train:-------",names_of_predictors)

names_of_predictors = list(Y_train.columns.values)
print("names of labels Y train-------",names_of_predictors)


names_of_predictors = list(X_test.columns.values)
print("names of predictors in X train:-------",names_of_predictors)
names_of_labels = list(Y_test)
print("names of test labels in Y test:-------",names_of_labels)

#modifying scale of data (minMax and robust scale)
feature_list=X_train.columns
x=X_train[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_train.index)
X_train[feature_list]=df_temp
x=X_test[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_test.index)
X_test[feature_list]=df_temp

from imblearn.combine import SMOTETomek
undersample=SMOTETomek(random_state=49, sampling_strategy='minority')

Xtrain_smotetomek, Ytrain_smotetomek = undersample.fit_sample(X_train, Y_train)
model_LR = LogisticRegression(max_iter=4000,random_state=49,n_jobs=1, C=0.001, solver='saga',penalty='l1', class_weight='balanced') # for liblinear n_jobs is +1

model_rf = RandomForestClassifier(class_weight='balanced', random_state=49, bootstrap=True, max_depth=80, max_features='sqrt', min_samples_leaf=3, min_samples_split=12, n_estimators=1000)

model_svm= svm.SVC(class_weight='balanced', C=1, gamma=0.001, kernel='linear', probability=True)
model_GB= GradientBoostingClassifier(criterion= 'friedman_mse', learning_rate=0.15, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=1000)
model_knn=KNeighborsClassifier(n_neighbors=500,n_jobs=-1)
estimator=[]
estimator.append(('LR',model_LR))
# estimator.append(('RF',model_rf))
estimator.append(('svm',model_svm))
# estimator.append(('GB',model_GB))
# estimator.append(('knn',model_knn))

vot_hard = VotingClassifier(estimators=estimator, voting='hard')
vot_hard.fit(Xtrain_smotetomek, Ytrain_smotetomek.values.ravel())
predictions = vot_hard.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score

print(classification_report(Y_test,predictions))
print("Balanced Accuracy from hard voting:",balanced_accuracy_score(Y_test, predictions))

plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])

# Voting Classifier with soft voting
vot_soft = VotingClassifier(estimators=estimator, voting='soft')
vot_soft.fit(Xtrain_smotetomek, Ytrain_smotetomek.values.ravel())
predictions = vot_soft.predict(X_test)

print(classification_report(Y_test,predictions))
print("Balanced Accuracy from soft voting:",balanced_accuracy_score(Y_test, predictions))

plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
# ------------------------------stacking technique----------------------------------------------
estimator_stack = [model_LR,model_rf, model_svm]
S_train, S_test = stacking(estimator_stack,
                           Xtrain_smotetomek, Ytrain_smotetomek.values.ravel(), X_test,
                           regression=False,

                           mode='oof_pred_bag',

                           needs_proba=False,

                           save_dir=None,

                           metric=accuracy_score,

                           n_folds=4,

                           stratified=True,

                           shuffle=True,

                           random_state=0,

                           verbose=2)


from xgboost import XGBClassifier

model = XGBClassifier(random_state=0, n_jobs=-1, learning_rate=0.1,
                      n_estimators=100, max_depth=3)
model=AdaBoostClassifier(n_estimators=50, base_estimator=model_svm,learning_rate=1)

model = model_GB.fit(S_train, Ytrain_smotetomek.values.ravel())
predictions = model.predict(S_test)
print('Final prediction score: [%.8f]' % accuracy_score(Y_test, predictions))
print("Balanced Accuracy",balanced_accuracy_score(Y_test, predictions))
print(classification_report(Y_test,predictions))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])

