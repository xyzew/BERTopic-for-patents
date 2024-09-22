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
from octis.evaluation_metrics.diversity_metrics import TopicDiversity

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

# Reduce stop words
ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)
# Create your representation model
representation_model = KeyBERTInspired()

# Cluster documents
# min_samples默认等于min_cluster_size
hdbscan_model = HDBSCAN(min_cluster_size=80, metric='euclidean', cluster_selection_method='eom',
                        prediction_data=True)

# # kmeans
# cluster_model = KMeans(n_clusters=80)
#Create model
topic_model = BERTopic(
    # Pipeline models
    embedding_model=embedding_model,
    umap_model=umap_model,
    hdbscan_model=hdbscan_model,
    vectorizer_model=vectorizer_model,
    # representation_model=representation_model,
    # Hyperparameters
    top_n_words=20,
    # If verbose=True is set when training a BERTopic model, you may see various log information during the model training process.
    verbose=True,
    calculate_probabilities=True
    ,ctfidf_model=ctfidf_model #Reduce stop words
    ,representation_model=representation_model#Reduce stop words
    ,nr_topics=21
)

topics,probility=topic_model.fit_transform(abstracts, embeddings)

# # # Reduce outliers with pre-calculate embeddings instead Merge outliers
# new_topics = topic_model.reduce_outliers(abstracts, topics,probabilities=probility, strategy="probabilities")
# topic_model.update_topics(abstracts, topics=new_topics)



#
# # Calculate the similarity mean and variance
# similarity_map = topic_model.visualize_heatmap()
# b = similarity_map.data[0].__getattribute__("z")
# np_matrix = np.array(b)
# # Calculate the mask of the lower triangular matrix, the diagonal is also excluded
# lower_triangle_mask = np.tril(np_matrix, -1)
# # Extract the non-zero elements of the lower triangle part
# lower_triangle_elements = lower_triangle_mask[lower_triangle_mask != 0]
# # Calculate the mean
# mean_value = np.mean(lower_triangle_elements)
# # Calculate variance
# variance_value = np.var(lower_triangle_elements)
#
#
# # Calculate consistency score
# topics = topic_model.topics_
# topics2 = topic_model.get_topic_info()['Topic'].values
# documents = pd.DataFrame({"Document": abstracts,
#                           "ID": range(len(abstracts)),
#                           "Topic": topics})
# documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
# cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
# vectorizer = topic_model.vectorizer_model
# analyzer = vectorizer.build_analyzer()
# words = vectorizer.get_feature_names_out()
# tokens = [analyzer(doc) for doc in cleaned_docs]
# dictionary = corpora.Dictionary(tokens)
# corpus = [dictionary.doc2bow(token) for token in tokens]
# topic_words = [[words for words, _ in topic_model.get_topic(topic)]
#                for topic in topics2]
# coherence_model = CoherenceModel(topics=topic_words,
#                                  texts=tokens,
#                                  corpus=corpus,
#                                  dictionary=dictionary,
#                                  coherence='c_v')
# coherence = coherence_model.get_coherence()
# print(mean_value)
# print(variance_value)
# print(coherence)

# # Calculate topic difference
# metric = TopicDiversity(topk=10) # Initialize metric
# topic_diversity_score = metric.score(topic_model) # Compute score of the metric


# # Merge topics
# topics_to_merge = [11, 15]
# topic_model.merge_topics(abstracts, topics_to_merge)
#
#
# Assume 'topic_model' is a trained topic model
# Save visualization as HTML file
topic_visualization = topic_model.visualize_topics()
topic_visualization.write_html("D:/python/Bertopic/可视化结果/topics_visualizationF.html")

#
#
#
similarity_map=topic_model.visualize_heatmap()
# # Save heatmap as interactive HTML file
similarity_map.write_html("D:/python/Bertopic/可视化结果/similarity_heatmapF.html")
#
#
#
# Bar chart
barcahrt=topic_model.visualize_barchart(top_n_topics=30,width=250)
barcahrt.write_html("D:/python/Bertopic/可视化结果/barcahrtF.html")
# #
# #
# topics = topic_model.topics_
# data['Topic']=topics
# data.to_excel("ResultVS H.xlsx", index=False)

#
#
# # # dynamic topic model
# timestamps= data['priority date']
# timestamps=timestamps.tolist()
# topics_over_time = topic_model.topics_over_time(abstracts, timestamps,nr_bins=30,datetime_format="%Y-%m-%d")
# DynamicPicture=topic_model.visualize_topics_over_time(topics_over_time)
# DynamicPicture.write_html("D:/python/Bertopic/可视化结果/DynamicPictureF.html")



# if __name__ == '__main__':
#     main()