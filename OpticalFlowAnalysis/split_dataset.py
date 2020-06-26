
# split dataset into train, test, and validation sets.

import pandas as pd

file = "FinalDatasetForSplit.csv"

trainVideos = ["錄製_2019_11_23_16_16_02_506.mp4", "錄製_2019_11_29_16_14_59_189.mp4",
               "錄製_2019_11_24_16_10_26_600.mp4", "錄製_2019_11_20_10_13_54_900.mp4",
               "錄製_2019_11_28_09_42_38_725.mp4", "錄製_2019_11_28_07_43_03_380.mp4"]

validationVideos = ["錄製_2019_11_28_12_05_07_124.mp4", "錄製_2019_11_29_08_23_34_390.mp4",
                    "錄製_2019_11_28_16_11_37_783.mp4", "錄製_2019_11_21_07_25_55_214.mp4",
                    "錄製_2019_11_20_11_12_52_295.mp4", "錄製_2019_11_25_12_24_03_891.mp4"]

testVideos = ["錄製_2019_11_28_12_59_28_589.mp4", "錄製_2019_11_28_11_02_40_268.mp4",
              "錄製_2019_11_29_13_07_47_434.mp4"]

df = pd.read_csv(file, names=["filename", "framenumber", "x0", "y0", "x1", "y1",
                              "label","velocity1","velocity2", "kmeans ","HDBScan","filepath" ,"comment1", "comment2", "comment3",
                              "comment4"], engine="python" ,encoding='utf-8-sig')
# remove all comments in dataset
df = df.drop(["comment1", "comment2", "comment3", "comment4"], axis=1)
# remove absoulte filepath
df.filename.replace(r"E\:\\.{40,}\\", "", regex=True, inplace=True)

# train
trainSet = df[df["filename"].isin(trainVideos)]
lengthTrain = len(trainSet.index)
#print(len(trainSet.loc[trainSet["label"] > 0].index))
print("Trainset==0: ",len(trainSet.loc[trainSet["label"] == 0].index))
print(lengthTrain)
assert lengthTrain == 13290

# validation
validSet = df[df["filename"].isin(validationVideos)]
#print(len(validSet.loc[validSet["label"] > 0].index))
print("ValidateSet==0:",len(validSet.loc[validSet["label"] == 0].index))

lengthValid = len(validSet.index)
assert lengthValid == 3451
print(lengthValid)

# test
testSet = df[df["filename"].isin(testVideos)]
#print(len(testSet.loc[testSet["label"] > 0].index))
print("TestSet==0:",len(testSet.loc[testSet["label"] == 0].index))

lengthTest = len(testSet.index)
assert lengthTest == 3454
print(lengthTest)
assert (lengthTest + lengthValid + lengthTrain) == 20195

# write out without header or index column
trainSet.to_csv("TrainingData/train_multiclass.csv", index=False, header=False, encoding='utf_8_sig')
validSet.to_csv("TrainingData/valid_multiclass.csv", index=False, header=False,encoding='utf_8_sig')
testSet.to_csv("TrainingData/test_multiclass.csv", index=False, header=False, encoding='utf_8_sig')
