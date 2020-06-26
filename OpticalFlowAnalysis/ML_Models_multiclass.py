import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import pyplot
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.utils import shuffle
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import cross_val_score, GridSearchCV

from sklearn.preprocessing import minmax_scale
from sklearn.decomposition import PCA
from sklearn import svm
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


train = pd.read_csv("TrainingData/train_multiclass.csv")
test = pd.read_csv("TrainingData/valid_multiclass.csv")

train=shuffle(train,random_state=49)
train.reset_index(drop=True,inplace=True)
test=shuffle(test,random_state=49)
test.reset_index(drop=True,inplace=True)
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
#Frequency Distribution of the Outome
train_outcome = pd.crosstab(index=train["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print(train_outcome)
test_outcome = pd.crosstab(index=test["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print(test_outcome)
#Visualizing Outcome Distribution
temp = train["label"].value_counts()
df = pd.DataFrame({'labels': temp.index,
                   'values': temp.values
                  })

#df.plot(kind='pie',labels='labels',values='values', title='Activity Ditribution',subplots= "True")

labels = df['labels']
sizes = df['values']
colors = ['yellowgreen', 'gold', 'lightskyblue', 'pink','cyan','black']
patches, texts = plt.pie(sizes, colors=colors, shadow=True, startangle=90, pctdistance=1.1, labeldistance=1.2)
plt.legend(patches, labels, loc="best")
plt.axis('equal')
plt.tight_layout()
plt.show()

# Seperating Predictors and Outcome values from train and test sets


X_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label',"velocity_meters", "hdbscan","path"],axis=1))
Y_train =pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity_pixels", "velocity_meters","kmeans", "hdbscan","path"],axis=1))
X_test = pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label',"velocity_meters", "hdbscan","path"],axis=1))
Y_test =pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity_pixels", "velocity_meters","kmeans", "hdbscan","path"],axis=1))
print("Train x and y", X_train, Y_train)
print("Test x and y", X_test, Y_test)
num_cols = X_train._get_numeric_data().columns
print("Number of numeric features in X_train:",num_cols.size)
num_cols = Y_train._get_numeric_data().columns
print("Number of numeric features in Y_train:",num_cols.size)

num_cols = X_test._get_numeric_data().columns
print("Number of numeric features in X_test:",num_cols.size)
num_cols = Y_test._get_numeric_data().columns
print("Number of numeric features in Y_test:",num_cols.size)

names_of_predictors = list(X_train.columns.values)
print("names of predictors:-------",names_of_predictors)



names_of_predictors = list(Y_train.columns.values)
print("names of labels-------",names_of_predictors)


num_cols = X_train._get_numeric_data().columns
print("Number of numeric features:",num_cols.size)

#list(set(X_train.columns) - set(num_cols))

names_of_predictors = list(X_train.columns.values)
print("names of predictors:-------",names_of_predictors)
names_of_labels = list(Y_test)
print("names of test labels:-------",names_of_labels)

names_of_labels = list(Y_train)
print("names of train labels:-------",names_of_labels)
print("X_train befor scale:---",X_train)
#modifying scale of data (minMax)
feature_list=X_train.columns
x=X_train[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_train.index)
X_train[feature_list]=df_temp
print("x train after scale",X_train)

x=X_test[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_test.index)
X_test[feature_list]=df_temp

print("x test after scale",X_test)

from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
#MLP Model-----------------------------------------------------
param={"activation": ['identity', 'logistic', 'tanh', 'relu'], "alpha": [0.0001, 0.001, 0.01, 0.1,1], "learning_rate_init": [0.0001, 0.001, 0.01, 0.1,1] , "solver":['lbfgs', 'sgd', 'adam']}

mlp = MLPClassifier(random_state=49, hidden_layer_sizes=(8,4), max_iter=50, activation='tanh', alpha=0.0001, learning_rate_init= 0.01)
mlp=MLPClassifier(random_state=49,  max_iter=1000)
mlp_model = GridSearchCV(mlp, param, cv=3)
mlp_model.fit(X_train, Y_train.values.ravel())
predictions = mlp_model.predict(X_test)
#print("Best Parameters:-------------",mlp_model.best_params_)
#print(mlp_model.best_score_)# predictions = mlp.predict(X_test)
#using big value of alpha to panalize the error
# mlp = MLPClassifier(activation='relu',random_state=49,alpha=0.001, learning_rate_init=0.001)
# mlp.fit(X_train, Y_train.values.ravel())
#LR Model-----------------------------------------------------
LR=LogisticRegression(solver = 'lbfgs', max_iter=500, class_weight='balanced', random_state=49)
LR.fit(X_train,Y_train.values.ravel())
predictions = LR.predict(X_test)

#SVM Model-----------------------------------------------------
parameters = {'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],'C': [1, 10, 100, 1000]}#-----hyperparameter tuning
#reference: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74
svm_model = GridSearchCV(svm.SVC(class_weight='balanced'),parameters,cv=3)
svm_model.fit(X_train, Y_train.values.ravel())
predictions = svm_model.predict(X_test)
#Random Forest---------------------------------------------------------------------
parameters_RF=param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['sqrt'],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestClassifier(class_weight='balanced', random_state=49,  bootstrap=True, max_depth= 80, max_features='auto' ,min_samples_leaf=3, min_samples_split=8, n_estimators=1000)
# model_rf=GridSearchCV(rf,parameters_RF,cv=3)
rf.fit(X_train, Y_train.values.ravel())
predictions = rf.predict(X_test)
from sklearn.metrics import classification_report, confusion_matrix, balanced_accuracy_score
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))
print(balanced_accuracy_score(Y_test, predictions))
plot_confusion_matrix(confusion_matrix(Y_test,predictions),['Dolphin','Non-Dolphin'])
