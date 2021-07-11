import numpy as np
from sklearn import linear_model
from sklearn import metrics
from pandas import read_csv
from time import perf_counter


#train = read_csv('EHR.csv', usecols = ['Attestation_Year','Payment_Year'])
#train = read_csv('pvch.csv', usecols = ['ZIP Code','Payment_footnote'])
train = read_csv('BRFSS.csv', usecols = ['Sample_Size','Confidence_limit_Low'])
#data = train[['Attestation_Year','Payment_Year']]
#data = train[['ZIP Code','Payment_footnote']]
data = train[['Sample_Size','Confidence_limit_Low']]
#original_DS = train.Payment_Year

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

clf = linear_model.SGDRegressor(penalty='elasticnet',alpha=0.0005,l1_ratio=0.2)
clf.fit(x_train, y_train)
predicted = clf.predict(x_test)
#train.Payment_Year[train.Payment_Year.isnull()] = predicted
#train.Payment_footnote[train.Payment_footnote.isnull()] = predicted
train.Confidence_limit_Low[train.Confidence_limit_Low.isnull()] = predicted
print(train.Confidence_limit_Low)
#print(train.Payment_Year)
#print(train.Payment_footnote)

'''
Section for applying outlier metrics for plausability evaluation
By using z-score
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


#detecting outliers in imputed dataset
time_start = perf_counter()
#outlier_datapoints = detect_outlier(train.Payment_Year)
#outlier_datapoints = detect_outlier(train.Payment_footnote)
outlier_datapoints = detect_outlier(train.Confidence_limit_Low)
print(outlier_datapoints)
time_stop = perf_counter() 
print("Elapsed time:", time_stop - time_start)


