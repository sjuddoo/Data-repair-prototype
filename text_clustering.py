from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from time import perf_counter

#time_start = time.clock()
time_start = perf_counter()

from pandas import read_csv

#df = read_csv('EHR.csv', usecols = ['Specialty'])
df = read_csv('BRFSS.csv', usecols = ['Class'])

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

time_stop = perf_counter() 
print("Elapsed time:", time_stop - time_start)
#time_elapsed = (time.clock() - time_start)
#print(time_elapsed)
