from Tkinter import *
#import Tkinter.messagebox

def LinearRegression():
    from sklearn.linear_model import LinearRegression
    import time
    import numpy as np
    from sklearn import metrics
    from pandas import read_csv

    time_start = time.clock()

    train = read_csv('EHR.csv', usecols = ['Attestation_Year','Payment_Year'])
    linreg = LinearRegression()
    data = train[['Attestation_Year','Payment_Year']]
    original_DS = train.Payment_Year


    #Step-1: Split the dataset that contains the missing values and no missing values are test and train respectively.

    x_train = data[data['Payment_Year'].notnull()].drop('Payment_Year', axis= 1)
    y_train = data[data['Payment_Year'].notnull()]['Payment_Year']
    x_test = data[data['Payment_Year'].isnull()].drop('Payment_Year', axis=1)
    y_test = data[data['Payment_Year'].isnull()]['Payment_Year']

    #Step-2: Train the machine learning algorithm

    linreg.fit(x_train, y_train)

    #Step-3: Predict the missing values in the attribute of the test data.

    predicted = linreg.predict(x_test)

    #Step-4: Let’s obtain the complete dataset by combining with the target attribute.

    train.Payment_Year[train.Payment_Year.isnull()] = predicted
    #print(train.Payment_Year) #getting the imputed data frame

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
    #outlier_datapoints = detect_outlier(original_DS)
    #detecting outliers in imputed dataset
    outlier_datapoints = detect_outlier(train.Payment_Year)
    print(outlier_datapoints)
    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)



def Clustering():
    from sklearn.cluster import KMeans
    from sklearn.metrics import adjusted_rand_score
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    import time

    time_start = time.clock()

    from pandas import read_csv

    df = read_csv('EHR.csv', usecols = ['Program_Type'])

    documents = str(df.dropna().values.tolist()).split(",")

    tfidf_vectorizer=TfidfVectorizer(use_idf=True, stop_words = 'english')
     
    # just send in all your docs here
    tfidf_vectorizer_vectors=tfidf_vectorizer.fit_transform(documents)
    first_vector_tfidfvectorizer=tfidf_vectorizer_vectors[0]
     
    # place tf-idf values in a pandas data frame
    df1 = pd.DataFrame(first_vector_tfidfvectorizer.T.todense(), index=tfidf_vectorizer.get_feature_names(), columns=["tfidf"])

    #with pd.option_context('display.max_rows', None, 'display.max_columns', None):print(df1)
    true_k = 1
    model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
    model.fit(tfidf_vectorizer_vectors)

    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf_vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :100]:
            print(' %s' % terms[ind]),
        print

    time_elapsed = (time.clock() - time_start)
    print(time_elapsed)





app = Tk()
app.title("Data Quality prototype")
app.geometry('500x500')

labelText = StringVar()
labelText.set("Proceed with missing value imputation first, then cater for noise")
label1 =  Label(app, textvariable = labelText, height = 4)
label1.pack()

button1 = Button(app, text = 'Impute with Linear Regression', width = 40,command=LinearRegression)
button1.pack()
button2 = Button(app, text = 'Noise detection', width = 40,command=Clustering)
button2.pack()
app.mainloop()

