import pandas as pd
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from bertopic.vectorizers import ClassTfidfTransformer
from bertopic.representation import KeyBERTInspired
from sklearn.cluster import KMeans
import numpy as np


# def main():
data = pd.read_excel("Result4.xlsx")  # content type
abstracts = data['content_cutted']

# Precompute embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = np.load("D:/python/Bertopic/embeddings/embeddings.npy")
# Dimensionally reduce embeddings
umap_model = UMAP(n_neighbors=15, n_components=60, min_dist=0.01, metric='cosine', random_state=42)

# Text matrix
vectorizer_model = CountVectorizer(analyzer='word',
                                   min_df=1,  # minimum reqd occurences of a word
                                   stop_words='english',  # remove stop words
                                   lowercase=True,  # convert all words to lowercase
                                   token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                   max_features=4000,  # max number of uniq words
                                   ngram_range=(1,3)
                                   )

## Reduce stop words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# Create your representation model
representation_model = KeyBERTInspired()


# Cluster documents
# min_samples默认等于min_cluster_size
# hdbscan_model = HDBSCAN(min_cluster_size=80, metric='euclidean', cluster_selection_method='eom',
#                         prediction_data=True)

# # kmeans
cluster_model = KMeans(n_clusters=21)

#Create model
topic_model = BERTopic(
    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=cluster_model,
    vectorizer_model=vectorizer_model,
    # representation_model=representation_model,
    # Hyperparameters
    top_n_words=20,
    # If verbose=True is set when training a BERTopic model, you may see various log information during the model training process.
    verbose=True,
    calculate_probabilities=True
    ,ctfidf_model=ctfidf_model #reduce stop words
    ,representation_model=representation_model#reduce stop words
    # ,nr_topics=21
)

topics,probility=topic_model.fit_transform(abstracts, embeddings)

# Reduce outliers with pre-calculate embeddings instead merge outliers
# new_topics = topic_model.reduce_outliers(abstracts, topics,strategy="embeddings")
# topic_model.update_topics(abstracts, topics=new_topics)

# # Merge topics
# topics_to_merge = [[21, 16],[19,7],[20,12]]
# topic_model.merge_topics(abstracts, topics_to_merge)

# Assume 'topic_model' is a trained topic model
topic_visualization = topic_model.visualize_topics()
# Save visualization as HTML file
topic_visualization.write_html("D:/python/Bertopic/可视化结果/topics_visualizationK.html")


similarity_map=topic_model.visualize_heatmap()
# Save the heat map as an interactive HTML file
similarity_map.write_html("D:/python/Bertopic/可视化结果/similarity_heatmapK.html")

# Bar chart
barcahrt=topic_model.visualize_barchart(top_n_topics=30,width=500)
barcahrt.write_html("D:/python/Bertopic/可视化结果/barcahrtK.html")

#Save results to excel
topics = topic_model.topics_
data['Topic']=topics
data.to_excel("ResultVS K.xlsx", index=False)


# # Dynamic topic model
# timestamps= data['priority date']
# timestamps=timestamps.tolist()
# topics_over_time = topic_model.topics_over_time(abstracts, timestamps,nr_bins=30,datetime_format="%Y-%m-%d")
# DynamicPicture=topic_model.visualize_topics_over_time(topics_over_time)
# DynamicPicture.write_html("D:/python/Bertopic/可视化结果/DynamicPictureK.html")


#
# if __name__ == '__main__':
#     main()