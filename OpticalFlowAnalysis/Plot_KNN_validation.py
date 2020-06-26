import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier
from imblearn.combine import SMOTETomek
from sklearn.utils import shuffle
from sklearn.metrics import  balanced_accuracy_score
from sklearn.preprocessing import minmax_scale, robust_scale

train = pd.read_csv("TrainingData/datawithclusters/train_clusters.csv")
valid = pd.read_csv("TrainingData/datawithclusters/valid_clusters.csv")
test = pd.read_csv("TrainingData/datawithclusters/test_clusters.csv")
train=shuffle(train,random_state=49)
train.reset_index(drop=True,inplace=True)
valid=shuffle(valid,random_state=49)
valid.reset_index(drop=True,inplace=True)
test=shuffle(test,random_state=49)
test.reset_index(drop=True,inplace=True)
print("Any missing sample in training set:",train.isnull().values.any())
print("Any missing sample in valid set:",valid.isnull().values.any(), "\n")
print("Any missing sample in test set:",test.isnull().values.any(), "\n")
train_outcome = pd.crosstab(index=train["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print("Train:",train_outcome)
valid_outcome = pd.crosstab(index=valid["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print("Valid",valid_outcome)
test_outcome = pd.crosstab(index=test["label"],  # Make a crosstab
                              columns="count")      # Name the count column
print(test_outcome)

X_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label', "kmeans","path"],axis=1))
Y_train = pd.DataFrame(train.drop(["filename", "framenumber", "x0", "y0", "x1", "y1", "velocity","kmeans", "hdbscan","path"],axis=1))
X_valid = pd.DataFrame(valid.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label', "kmeans","path"],axis=1))
Y_valid =pd.DataFrame(valid.drop(["filename", "framenumber", "x0", "y0", "x1", "y1", "velocity","kmeans", "hdbscan","path"],axis=1))
X_test= pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1",'label', "kmeans","path"],axis=1))
Y_test =pd.DataFrame(test.drop(["filename", "framenumber", "x0", "y0", "x1", "y1","velocity_pixels", "velocity","kmeans", "hdbscan","path"],axis=1))

#modifying scale of data (minMax and robust scale)
feature_list=X_train.columns
x=X_train[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_train.index)
X_train[feature_list]=df_temp

x=X_valid[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_valid.index)
X_valid[feature_list]=df_temp

x=X_test[feature_list].values
x_scaled=minmax_scale(x)
df_temp=pd.DataFrame(x_scaled, columns=feature_list,index=X_test.index)
X_test[feature_list]=df_temp

sample=SMOTETomek(random_state=49, sampling_strategy='minority')
Xtrain_sample, Ytrain_sample= sample.fit_sample(X_train, Y_train)
knn = KNeighborsClassifier(n_neighbors=200,n_jobs=-1)
neighoiurs = [i for i in range(20, 220, 20)]
list_of_TrainAccuracy = []
list_of_validAccuracy = []
list_of_testAccuracy = []
for n_neighbour in neighoiurs:
    knn = KNeighborsClassifier(n_neighbors= n_neighbour, n_jobs=-1)
    knn.fit(Xtrain_sample, Ytrain_sample.values.ravel())
    predictions_train = knn.predict(X_train)
    predictions_valid=knn.predict(X_valid)
    predictions_test=knn.predict(X_test)
    list_of_TrainAccuracy.append(balanced_accuracy_score(Y_train, predictions_train))
    list_of_validAccuracy.append(balanced_accuracy_score(Y_valid, predictions_valid))
    list_of_testAccuracy.append(balanced_accuracy_score(Y_test, predictions_test))

# Plot mean accuracy scores for training on validation and test sets

plt.plot(neighoiurs, list_of_validAccuracy, marker='', color='red', label='Validation Accuracy', linewidth=2)
plt.plot(neighoiurs, list_of_testAccuracy, marker='', color='blue', label='Test Accuracy',linewidth=2)
# Create plot
plt.title("Validation Curve With KNN")
plt.xlabel("parameters")
plt.ylabel("Balanced Accuracy Score")
import matplotlib.patches as mpatches
valid = mpatches.Patch(color='red', label='Validation Accuracy')
test = mpatches.Patch(color='blue', label='Test Accuracy')
plt.legend(handles=[test, valid])

plt.tight_layout()
plt.show()
