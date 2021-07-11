from sklearn.linear_model import LinearRegression
from time import perf_counter
import numpy as np
from sklearn import metrics
from pandas import read_csv

#time_start = time.clock()

#train = read_csv('EHR.csv', usecols = ['Attestation_Year','Payment_Year'])#1st dataset
#train = read_csv('pvch.csv', usecols = ['ZIP Code','Payment_footnote'])#2nd dataset
train = read_csv('BRFSS.csv', usecols = ['Sample_Size','Confidence_limit_Low'])
linreg = LinearRegression()
#data = train[['Attestation_Year','Payment_Year']]
#data = train[['ZIP Code','Payment_footnote']]
data = train[['Sample_Size','Confidence_limit_Low']]
#original_DS = train.Payment_Year
#original_DS = train.Payment_footnote
original_DS = train.Confidence_limit_Low

#Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.
'''
x_train = data[data['Payment_Year'].notnull()].drop('Payment_Year', axis= 1)
y_train = data[data['Payment_Year'].notnull()]['Payment_Year']
x_test = data[data['Payment_Year'].isnull()].drop('Payment_Year', axis=1)
y_test = data[data['Payment_Year'].isnull()]['Payment_Year']

x_train = data[data['Payment_footnote'].notnull()].drop('Payment_footnote', axis= 1)
y_train = data[data['Payment_footnote'].notnull()]['Payment_footnote']
x_test = data[data['Payment_footnote'].isnull()].drop('Payment_footnote', axis=1)
y_test = data[data['Payment_footnote'].isnull()]['Payment_footnote']
'''
x_train = data[data['Confidence_limit_Low'].notnull()].drop('Confidence_limit_Low', axis= 1)
y_train = data[data['Confidence_limit_Low'].notnull()]['Confidence_limit_Low']
x_test = data[data['Confidence_limit_Low'].isnull()].drop('Confidence_limit_Low', axis=1)
y_test = data[data['Confidence_limit_Low'].isnull()]['Confidence_limit_Low']

#Step-2: Train the machine learning algorithm

linreg.fit(x_train, y_train)

#Step-3: Predict the missing values in the attribute of the test data.

predicted = linreg.predict(x_test)

#Step-4: Let’s obtain the complete dataset by combining with the target attribute.

#train.Payment_Year[train.Payment_Year.isnull()] = predicted
#train.Payment_footnote[train.Payment_footnote.isnull()] = predicted
train.Confidence_limit_Low[train.Confidence_limit_Low.isnull()] = predicted
print(train.Confidence_limit_Low) #getting the imputed data frame

'''
Section for applying outlier metrics for plausability evaluation
'''

outliers=[]
def detect_outlier(data_1):
    
    threshold=3
    mean_1 = np.mean(data_1)
    std_1 =np.std(data_1)
    count = 0
    
    for y in data_1:
        z_score= (y - mean_1)/std_1 
        if np.abs(z_score) > threshold:
            #outliers.append(y)
            count = count + 1
    return count

# detecting outliers in original dataset
outlier_datapoints = detect_outlier(original_DS)
#detecting outliers in imputed dataset

time_start = perf_counter()
#outlier_datapoints = detect_outlier(train.Payment_Year)
#outlier_datapoints = detect_outlier(train.Payment_footnote)
outlier_datapoints = detect_outlier(train.Confidence_limit_Low)
print(outlier_datapoints)

time_stop = perf_counter() 
print("Elapsed time:", time_stop - time_start)

