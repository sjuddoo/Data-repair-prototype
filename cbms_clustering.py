from sklearn.cluster import KMeans
import numpy as np
from time import perf_counter
#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split
from pandas import read_csv
from sklearn.neighbors import KNeighborsClassifier
#train = read_csv('EHR.csv', usecols = ['Attestation_Year','Payment_Year'])
train = read_csv('pvch.csv', usecols = ['ZIP Code','Payment_footnote'])
#data = train[['Attestation_Year','Payment_Year']]
data = train[['ZIP Code','Payment_footnote']]

'''
x_train = data[data['Payment_Year'].notnull()].drop('Payment_Year', axis= 1)
y_train = data[data['Payment_Year'].notnull()]['Payment_Year']
x_test = data[data['Payment_Year'].isnull()].drop('Payment_Year', axis=1)
y_test = data[data['Payment_Year'].isnull()]['Payment_Year']
'''

x_train = data[data['Payment_footnote'].notnull()].drop('Payment_footnote', axis= 1)
y_train = data[data['Payment_footnote'].notnull()]['Payment_footnote']
x_test = data[data['Payment_footnote'].isnull()].drop('Payment_footnote', axis=1)
y_test = data[data['Payment_footnote'].isnull()]['Payment_footnote']

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

kmeans = KMeans(n_clusters=6, random_state=0).fit(x_train,y_train)


#need to get 10 list of clusters

labels = kmeans.labels_

for i in range(6):
    #for each cluster, apply KNN algo
    
    A = x_train[(labels == i)]
    B = y_train[(labels == i)]
    c = abs(B.count())
    #print(c)
    
    
    if (c > 2):
        A_train, A_test, B_train, B_test = train_test_split(A, B, test_size=0.3, random_state=1, stratify=B)

        # Create KNN classifier
        knn = KNeighborsClassifier(n_neighbors = 1, metric = 'correlation')

        # Fit the classifier to the data
        knn.fit(A_train,B_train)
        d = np.array(B_test).reshape(-1,1)
        knn.predict(d)
        #print(d)
        
        time_start = perf_counter()

        outlier_datapoints = detect_outlier(d)
        print(outlier_datapoints)

        time_stop = perf_counter() 
        print("Elapsed time:", time_stop - time_start)
       
       
    

