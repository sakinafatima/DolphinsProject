import pylab as pl
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, classification_report, recall_score, balanced_accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
#for drawing confusion matrix taken from https://stackoverflow.com/questions/19233771/sklearn-plot-confusion-matrix-with-labels
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
            plt.text(j, i, "{:1f}".format(cm[i, j], fontsize='x-large'),
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

train = pd.read_csv("TrainingData/datawithclusters/train_clusters.csv")
print("Shape of Train",train.shape)
train.head()
test = pd.read_csv("TrainingData/datawithclusters/valid_clusters.csv")
#uncomment below line for testing on test data
#test = pd.read_csv("TrainingData/datawithclusters/test_clusters.csv")
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
X_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label',"kmeans","path"],axis=1))
#X_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label', "hdbscan","path"],axis=1)) #uncomment for using kmeans clustering
Y_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity","kmeans", "hdbscan","path"],axis=1))
X_test = pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label',"kmeans","path"],axis=1))
#X_test = pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label', "hdbscan","path"],axis=1))#uncomment for using kmeans clustering
Y_test =pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity","kmeans", "hdbscan","path"],axis=1))
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
print("Balanced Accuracy: ",balanced_accuracy_score(Y_test, predictions))
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
# plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
# #2nd Test----------------------Logistic Regression--------------------------with over sampling Smote
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=49)

Xtrain_smote, ytrain_smote = smote.fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1) # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

LR_model.fit(Xtrain_smote,ytrain_smote.values.ravel())
#
# # get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print("Balanced Accuracy: ",balanced_accuracy_score(Y_test, predictions))

t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])

# # Third Test----------------------Logistic Regression--------------------------with balanced class weight without smote

# t0 = pl.time.time()
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

t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])

# #forth test----------------------Logestic Regression--------------------------------with undersampling Tomeklinks

from imblearn.under_sampling import TomekLinks
undersample = TomekLinks()
Xtrain_tomek, Ytrain_tomek = undersample.fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight='balanced') # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(Xtrain_tomek,Ytrain_tomek.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#fifth Test----------------------Logestic Regression--------------------------------with Smote-ENN
from imblearn.combine import SMOTEENN
combine = SMOTEENN(random_state=49, sampling_strategy='all')
Xtrain_combine, Ytrain_combine = combine.fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight="balanced") # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(Xtrain_combine,Ytrain_combine.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#sixth Test----------------------Logestic Regression--------------------------------with Smote-TomekLinks
from imblearn.combine import  SMOTETomek

combine = SMOTETomek(random_state=49 )
Xtrain_combine, Ytrain_combine = combine.fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight="balanced") # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(Xtrain_combine,Ytrain_combine.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
from imblearn.under_sampling import TomekLinks
undersample = TomekLinks()
Xtrain_tomek, Ytrain_tomek = undersample.fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight='balanced') # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(Xtrain_tomek,Ytrain_tomek.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)

print(f'Recall Logistic Regression {recall: .2f}')
print(report)
print(balanced_accuracy_score(Y_test, predictions))
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])



#seventh Test----------------------Logestic Regression--------------------------------undersampling with ENN
from imblearn.under_sampling import EditedNearestNeighbours
Xtrain_tomek, Ytrain_tomek = EditedNearestNeighbours().fit_sample(X_train, Y_train)
t0 = pl.time.time()
LR = LogisticRegression(max_iter=4000,
                            random_state=49,
                            n_jobs=1, class_weight='balanced') # for liblinear n_jobs is +1.

parameters = {"penalty": ['l1', 'l2'],'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], "solver":['liblinear','sag','saga']}

LR_model = GridSearchCV(LR, parameters, scoring="precision", cv=3)

# fit the classifier
LR_model.fit(Xtrain_tomek,Ytrain_tomek.values.ravel())

# get the prediction
predictions = LR_model.predict(X_test)

# model eval
recall = recall_score(Y_test,predictions)
report = classification_report(Y_test
                               ,predictions)
print(report)
print(balanced_accuracy_score(Y_test, predictions))
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters",LR_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#eighth Test----------------------Random Forest-------------------------------- class weight=balanced
t0 = pl.time.time()
parameters_RF=param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestClassifier(class_weight='balanced', random_state=49)
model_rf=GridSearchCV(rf,parameters_RF,cv=3)
model_rf.fit(X_train, Y_train.values.ravel())
predictions = model_rf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(balanced_accuracy_score(Y_test, predictions))
print("best parameters Random Forest",model_rf.best_params_)
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#ninth Test----------------------Random Forest-------------------------------- over sampling smote

from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy='minority', random_state=49)

Xtrain_smote, Ytrain_smote = smote.fit_sample(X_train, Y_train)
parameters_RF=param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
t0 = pl.time.time()
rf = RandomForestClassifier(class_weight='balanced', random_state=49, bootstrap= True, max_depth=80, max_features= 'sqrt', min_samples_leaf=3, min_samples_split=12, n_estimators=1000)
# model_rf=GridSearchCV(rf,parameters_RF,cv=3)

rf.fit(Xtrain_smote, Ytrain_smote.values.ravel())
predictions = rf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(balanced_accuracy_score(Y_test, predictions))
# print("best parameters Random Forest",rf.best_params_)
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#10th Test----------------------Random Forest-------------------------------- over sampling smote/ smotetomek and smoteTomek

from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
sample = SMOTETomek()

Xtrain_sample, Ytrain_sample= sample.fit_sample(X_train, Y_train)
parameters_RF=param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
t0 = pl.time.time()
rf = RandomForestClassifier(class_weight='balanced', random_state=49, bootstrap= True, max_depth=80, max_features= 'sqrt', min_samples_leaf=3, min_samples_split=12, n_estimators=1000)

# model_rf=GridSearchCV(rf,parameters_RF,cv=3)

rf.fit(X_train, Y_train.values.ravel())
predictions = rf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(balanced_accuracy_score(Y_test, predictions))
# print("best parameters Random Forest",model_rf.best_params_)
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#11th Test----------------------SVM-------------------------------- over sampling smote/ smotetomek and smoteTomek
parameters_SVM={'C': [1, 10, 100, 1000], 'kernel': ['linear','rbf', 'sigmoid'],'gamma': [0.001, 0.0001]}
t0 = pl.time.time()
from sklearn import svm
from imblearn.combine import SMOTETomek
sample=SMOTETomek(sampling_strategy='minority',random_state=49)
Xtrain_sample, Ytrain_sample= sample.fit_sample(X_train, Y_train)
# svm_model=svm.SVC(class_weight='balanced', C=1, gamma=0.001, kernel='linear')
svm_model = GridSearchCV(svm.SVC(class_weight='balanced'),parameters_SVM,cv=3)
svm_model.fit(X_train, Y_train.values.ravel())
predictions = svm_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print("Balanced Accuracy:",balanced_accuracy_score(Y_test, predictions))
# print("best parameters Random Forest",model_rf.best_params_)
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
print("best parameters:",svm_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#12th Test----------------------Gradient Boosting-------------------------------- simple
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.combine import SMOTETomek

# parameters={'max_depth':[2,3], 'learning_rate':[0.15,0.1,0.001,0.0001], 'n_estimators':[100,500,1000], 'min_samples_split':[2], 'min_samples_leaf':[1,3],  'criterion':['friedman_mse']}
t0 = pl.time.time()

sample=SMOTETomek(random_state=49, sampling_strategy='minority')
Xtrain_sample, Ytrain_sample= sample.fit_sample(X_train, Y_train)

GB_model=GradientBoostingClassifier(criterion= 'friedman_mse', learning_rate=0.15, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=1000)
# GB_model = GridSearchCV(GradientBoostingClassifier(),parameters,cv=3)

GB_model.fit(X_train, Y_train.values.ravel())
predictions = GB_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print("Balanced Accuracy:",balanced_accuracy_score(Y_test, predictions))
# print("best parameters Random Forest",model_rf.best_params_)
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
# print("best parameters:",GB_model.best_params_)
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#13th Test----------------------MLP-------------------------------- simple
from sklearn.neural_network import MLPClassifier
from imblearn.combine import SMOTETomek

param={"activation": ['identity', 'logistic', 'tanh', 'relu'], "alpha": [0.0001, 0.001, 0.01, 0.1,1], "learning_rate_init": [0.0001, 0.001, 0.01, 0.1,1] , "solver":['lbfgs', 'sgd', 'adam']}
t0 = pl.time.time()
#
sample=SMOTETomek(random_state=49, sampling_strategy='all')
Xtrain_sample, Ytrain_sample= sample.fit_sample(X_train, Y_train)
mlp_model = GridSearchCV(MLPClassifier(random_state=49, hidden_layer_sizes=(5,2), activation='identity', alpha=0.0001, learning_rate_init=0.0001, solver='lbfgs'),param,cv=3)
# mlp_model= MLPClassifier(random_state=49,  max_iter=5000, hidden_layer_sizes=(6,4), activation='relu', alpha=0.0001,  solver='lbfgs')
mlp_model.fit(X_train, Y_train.values.ravel())
predictions = mlp_model.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print("Balanced Accuracy:",balanced_accuracy_score(Y_test, predictions))
print("best parameters MLP",mlp_model.best_params_)
t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
#14th Test----------------------KNN-------------------------------- simple
from sklearn.neighbors import KNeighborsClassifier
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
#
t0 = pl.time.time()
# #
sample=SMOTETomek(random_state=49, sampling_strategy='minority')
Xtrain_sample, Ytrain_sample= sample.fit_sample(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors=200,n_jobs=-1)
# maximum on 200 neighbours with 0.7326 b.accuracy for hdbscan
# check neihbour knn for kmeans cluster? mix two classifiers
# n_jobs=-1 to utilize all cores
knn.fit(Xtrain_sample, Ytrain_sample.values.ravel())

predictions = knn.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print("Balanced Accuracy:",balanced_accuracy_score(Y_test, predictions))

t1 = pl.time.time() - t0
print("Time taken: {:.0f} min {:.0f} secs".format(*divmod(t1, 60)))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
