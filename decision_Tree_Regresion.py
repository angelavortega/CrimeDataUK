# Desicion Tree Regression

# Importing the libraries
from random import Random
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings

warnings.filterwarnings('ignore', category=UserWarning, module='openpyxl')

####################################### Changeable variables ############################################
ObservedData = False # True: predictions on the Observed Dataset
                    # Flase: predictions on the Self-Identified Dataset
randomSate = 30 # Choose any Int value. Use None if you want it to be fully random
test_size = .4     # Size of the test data: 0 < value < 1  
desiredIndVar = {   # True to the dependent variables that will be used in the model
    "Ethnicity": True,
    "Ageband": True,
    "Gender": True,
    "Country": True,
    "Region": True,
    "Policeforcename": True,
    "Court_Type": True,
    "Offencetype": True
}
desiredPredict = "sentenced"   # Options: "proceeded", "found_guilty", "sentenced", "tried_at_Crown_Court",
                            # "Absolute_discharge", "Conditional_Discharge", "fine", "Community_Sentence",
                            # "Suspended_Sentence", "Immediate_custody", "Otherwise_dealt_with"
saveMdlInf = False # True for saving the results and information of the models
##########################################################################################################

dictPredictions = {"proceeded": 8,
                "found_guilty": 9,
                "sentenced":	10,
                "tried_at_Crown_Court": 11,
                "Absolute_discharge": 12,
                "Conditional_Discharge": 13,
                "fine": 14,
                "Community_Sentence": 15,
                "Suspended_Sentence": 16,
                "Immediate_custody": 17,	
                "Otherwise_dealt_with": 18}
intPredict = dictPredictions[desiredPredict]
arrayIndVari = []
i = 1
for val in desiredIndVar.keys():
    if desiredIndVar[val]:
        if i < 8:
            arrayIndVari.append(i)
        else:
            arrayIndVari.append(20)
    i += 1
#print(arrayIndVari)
sheet_name = 'InputOb' if ObservedData else 'InputSelf'
csvStr = 'Ob' if ObservedData else 'Self'
dumiesTitle = 'dataValues/dumies_Ob.txt' if ObservedData else 'dataValues/dumies_Self.txt'
guiltyColum = 'found_guilty' if ObservedData else 'Found_Guilty'
procedAgaints = 'proceeded' if ObservedData else 'Proceeded_Against'
observed = 'data/defendants-observed-ethnic-appearance.csv'
selfIdent = 'data/defendants-self-identified-ethnicity.csv'
modelRsltPath = "Model_Results_{}.txt".format(csvStr)
path = observed if ObservedData else selfIdent
with open(dumiesTitle, 'w') as f:
    f.write("Dumies in files " + path)

# Check if txt of results exits
if not os.path.isfile(modelRsltPath):
    with open(modelRsltPath, 'w') as f:
        f.write("All models information and results are stored here\n")
# Importing the dataset
dataset = pd.read_csv(path)
#dataset = dataset[dataset[guiltyColum] > 0].reset_index(drop=True)
X = dataset.iloc[:, arrayIndVari]#.values
dataLen = len(X.columns)
flag = False
newX = None
orgColumNames = list(X.columns)
for i in range(0, dataLen):
    dumies = pd.get_dummies(X.iloc[:,[i]], prefix="", prefix_sep="")
    dumiesLen = len(dumies.columns) - 1
    # Dumies to TXT
    toFile = "\n{}: \n".format(orgColumNames[i])
    j = 0
    for val in list(dumies.columns):
        toFile = toFile + val + "\n" if j + 1 <= dumiesLen else toFile + val
        j += 1  
    with open(dumiesTitle, 'a') as f:
        f.write('\n' + toFile)
    if not flag:
        newX = dumies.iloc[:, :-1]
        flag = True
    else:
        j = 0
        for val in list(dumies.columns):
            if j < dumiesLen:
                #print(dumies.iloc[:, [j]])
                newX[val] = dumies.iloc[:, [j]].values  
            j += 1
#print(X)
#print(newX)
#newX[procedAgaints] = dataset.iloc[:,[8]]
newX.to_csv("dataValues/curateDataDefendats_{}.csv".format(csvStr),index=False)
X = newX.iloc[:, :].values
y = dataset.iloc[:, [intPredict]].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, random_state = randomSate)

# Training the Multiple Linear Regression model on the Training set
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
#np.set_printoptions(precision=2)
#print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
r2Score = r2_score(y_test, y_pred) 
rmse = mean_squared_error(y_test, y_pred, squared=False) 
mae = mean_absolute_error(y_test, y_pred)
print(path)
print(desiredIndVar)
print("Model R^2 Score: {}".format(r2Score))
print("Model RMSE: {}".format(rmse))
print("Model MAE: {}".format(mae))
inputData = pd.read_excel("input/inputSheet.xlsx", sheet_name=sheet_name)
inputData = inputData.iloc[:, :-1]
columnsNames = list(newX.columns)
pred_XData = np.zeros((len(inputData), len(columnsNames)))
for i in range(0, len(inputData)):
    for index in range(0, len(inputData.columns)): 
        value = inputData.iloc[i, index]
        if str(value) in columnsNames:
            j = columnsNames.index(value)
            pred_XData[i][j] = 1
y_inputPredict = regressor.predict(pred_XData)
inputData["pred_" + desiredPredict] = y_inputPredict
print(inputData)
inputData.to_csv("prediction/prediction_{}.csv".format(csvStr),index=False)
if not os.path.isfile("results.txt"):
    with open(modelRsltPath, 'w') as f:
        f.write('DataSource;R2;RMSE;MAE;Variables')

if saveMdlInf:
    with open("results.txt", 'a') as f:
        f.write('\n{};{};{};{};{}'.format(csvStr, r2Score, rmse, mae, desiredIndVar))
    with open(modelRsltPath, 'a') as f:
        f.write('\nData: {}'.format(path))
        f.write('\nRandom State: {}'.format(randomSate))
        f.write('\nTest Size: {}'.format(test_size))
        f.write('\nIndependent Variables: {}'.format(desiredIndVar))
        f.write('\nDependent Variables: {}'.format(desiredPredict))
        f.write("\nModel R^2 Score: {}".format(r2Score))
        f.write("\nModel RMSE: {}".format(rmse))
        f.write("\nModel MAE: {}\n".format(mae))