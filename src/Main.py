# Program focuses on classifying the previous pre-processed dataSet.
# From UCI datasets: http://archive.ics.uci.edu/ml/datasets/SPECTF+Heart
# A complete dataset, no empty vals, low dim, low numerosity and binary output.
# According to the Naive Bayes and K-Nearest Neighbor technique.
# The previous pre-processed dataSet, is the 6_reducedRanking.csv
# Which can be found in the data output of this repo:
# https://github.com/Davidmenamm/Data_Science_Data_Processing_1st_Step_Before_Machine_Learning/


# Imports
from Coordinator import Coordinator as Cd

# Other Files
# input
trainInputPath = r'data\input\2inputTrain.csv'
testInputPath = r'data\input\2inputTest.csv'

inputInitDataSet = r'data\input\4inputDataSet.csv'
# output
normOutputPath1 = r'data\output\1_normDataSet1.txt'
normOutputPath2 = r'data\output\1_normDataSet2.txt'
outputInitDataSet = r'data\output\4_initDataSet_norm.csv'


# function to write trazability of program to txt file
def writeToFile(data, path):
    data = str(data)
    with open(path, 'w') as f:  # open the file to write
        f.write(data)


# coordinator instance to carry out all data processing, process
coord = Cd()  # num of decimals as parameter

# Dataset completo, normalizado
normTrain = coord.normalize(inputInitDataSet)
normTrain.to_csv(
    outputInitDataSet, index=False)
# with open(path, 'w') as f:  # open the file to write
#     f.write(data)


# normalized dataSets (min-max)
normTrain = coord.normalize(trainInputPath)
normTest = coord.normalize(testInputPath)
writeToFile(normTrain, normOutputPath1)
writeToFile(normTest, normOutputPath2)

# type of metric
typeOfMetric = 2  # 1 manhattan, 2 euclidian

# get distances for one K
Kneighbors = 5
coord.getDistances(normTrain, normTest, typeOfMetric, Kneighbors)

# plot k vs acurracy
coord.plotK_Acurracy(normTrain, normTest, typeOfMetric)

# naive bayes
coord.runBayes(normTrain, normTest)
