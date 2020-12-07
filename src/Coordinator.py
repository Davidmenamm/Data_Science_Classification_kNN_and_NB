# In charge of administrating the program

# Imports
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
# from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import numpy as np
from sklearn.naive_bayes import GaussianNB


# Class Coordinator
class Coordinator:
    # constructor
    def __init__(self):
        self.distancesPath = r'data\output\distances.txt'
        self.bayesPath = r'data\output\nbayes.txt'

    # normalize columns (atributes), according to min-max technique
    def normalize(self, dataSetPath):
        # read dataset csv
        dataSet = pd.read_csv(dataSetPath, delimiter=',')

        # Calculate min and max of matrix
        min = 100  # temporal bigger number at first for min
        max = 0
        for _, dim in dataSet.iterrows():  # for dimension or attributes
            tempMin = dim.min()
            tempMax = dim.max()
            if tempMin < min and tempMin != 0:
                min = tempMin
            if tempMax > max:
                max = tempMax

        # numpy matrix to store temporarily
        rowNum = dataSet.shape[0]
        colNum = dataSet.shape[1]

        # apply min max normalization for all matrix
        countRows = 0
        npDataSet = np.zeros(shape=(rowNum, colNum))
        for _, row in dataSet.iterrows():
            # reindex from 0 to n-1
            countCols = 0
            for _, value in row.items():
                # columns to normalize
                if countCols > 0:
                    npDataSet[countRows, countCols] = round(
                        (value-min)/(max-min), 6)
                # avoid normalizing first column
                else:
                    npDataSet[countRows, countCols] = round(value, 6)
                # increment count cols
                countCols += 1
            # increment count rows
            countRows += 1
            # end when finishes
            if countRows == rowNum-1:
                break

        # turn numpy matrix to dataframe
        colNames = dataSet.columns.values.tolist()
        normDataSet = pd.DataFrame(data=npDataSet, columns=colNames)
        # return
        return normDataSet

    # to run the classification
    # features, target, num):
    def getDistances(self, trainDS, testDS, typeOfDist, selectedK):
        # from trainDStarget col and other cols, as numpy
        targTrain = trainDS.iloc[:, 0].to_numpy().astype(int)
        featTrain = trainDS.iloc[:, 1:].to_numpy()

        # from test remove binary target col, as numpy
        targTest = testDS.iloc[:, 0].to_numpy().astype(int)
        featTest = testDS.iloc[:, 1:].to_numpy()

        # in docs, if p=1 it is manhattan distance
        # and p=2 references euclidian distance
        knn = KNeighborsClassifier(n_neighbors=selectedK, p=typeOfDist)
        knn.fit(featTrain, targTrain)
        distances = knn.kneighbors(featTest, return_distance=True)

        # print to file
        with open(self.distancesPath, 'w') as f:  # open the file to write
            f.write(f'For k:{selectedK}\n')
            f.write(str(distances[0]))

    # to calculate best k
    def plotK_Acurracy(self, trainDS, testDS, typeOfDist):
        # from trainDStarget col and other cols, as numpy
        targTrain = trainDS.iloc[:, 0].to_numpy().astype(int)
        featTrain = trainDS.iloc[:, 1:].to_numpy()

        # from test remove binary target col, as numpy
        targTest = testDS.iloc[:, 0].to_numpy().astype(int)
        featTest = testDS.iloc[:, 1:].to_numpy()

        # TEST PRINTINGS
        print('trainDS\n', trainDS)
        print('featTrain\n', featTrain.shape)
        print('featTest\n', featTest)
        print('targTrain is\n', targTrain)

        # in docs, if p=1 it is manhattan distance
        # and p=2 references euclidian distance
        kValues = []
        acurracy = []
        print('\nk vs acurracy:')
        for k in range(1, 30):  # featTrain.shape[1]):
            # odd numbers from 1 to 30
            if k % 2 != 0:
                knn = KNeighborsClassifier(n_neighbors=k, p=typeOfDist)
                knn.fit(featTrain, targTrain)
                targ_pred = knn.predict(featTest)
                acc = metrics.accuracy_score(targTest, targ_pred)
                kValues.append(k)
                acurracy.append(acc)
                print(f'{k} -> {acc}')
        # matplot print
        kValuesNp = np.asarray(kValues, dtype=np.int32)
        acurracyNp = np.asarray(acurracy, dtype=np.float32)
        plt.plot(kValuesNp, acurracyNp)
        plt.xlabel('Value of K for KNN')
        plt.ylabel('Testing Accuracy')
        plt.show()

    # run naive bayes
    def runBayes(self, trainDS, testDS):
        # from trainDStarget col and other cols, as numpy
        targTrain = trainDS.iloc[:, 0].to_numpy().astype(int)
        featTrain = trainDS.iloc[:, 1:].to_numpy()

        # from test remove binary target col, as numpy
        targTest = testDS.iloc[:, 0].to_numpy().astype(int)
        featTest = testDS.iloc[:, 1:].to_numpy()

        # run the bayes
        nbc = GaussianNB()
        nbc.fit(featTrain, targTrain)
        # print results to file
        # print to file
        nbc_pred = nbc.predict(featTest)
        acc = metrics.accuracy_score(targTest, nbc_pred)
        with open(self.bayesPath, 'w') as f:  # open the file to write
            f.write(f'Naive Bayes Acurracy')
            f.write(str(acc))
            f.write('\nTarget Col Predicted:\n')
            f.write(str(nbc_pred))
            f.write('\nActual Column:\n')
            f.write(str(targTest))
