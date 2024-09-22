import pandas as pd
#Create an empty DataFrame to store all data
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


data = pd.read_excel("D:/python/Bertopic/K VS H/ZFinal.xlsx")  # content type
abstracts = data['content_cutted']
topicH=data['Topic']
topicH = topicH.tolist()

# Precompute embeddings
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# embeddings = embedding_model.encode(abstracts, show_progress_bar=True)
# np.save("D:/python/Bertopic/embeddings/embeddingsF.npy",embeddings)
embeddings = np.load("D:/python/Bertopic/embeddings/embeddingsF.npy")
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
# min_samples defaults to min_cluster_size
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
    top_n_words=10,
   # If verbose=True is set when training a BERTopic model, you may see various log information during the model training process.
    verbose=True
    ,ctfidf_model=ctfidf_model #Reduce stop words
    ,representation_model=representation_model#Reduce stop words
    ,nr_topics=20

)
topics=topic_model.fit_transform(abstracts, embeddings)
topic_model.update_topics(abstracts, topics=topicH,vectorizer_model=vectorizer_model)

# for item in range(0,20):
#     filename="D:/python/Bertopic/K VS H/" + "FAResultH" + str(item) + ".xlsx"
#     data = pd.read_excel(filename)
#     # 将数据追加到 all_data DataFrame 中
#     abstracts = data['content_cutted']
#     abstracts=abstracts.tolist()
#     timestamps = data['priority date']
#     timestamps = timestamps.tolist()
#     topics_over_time = topic_model.topics_over_time(abstracts, timestamps, nr_bins=30, datetime_format="%Y-%m-%d")
#     DynamicPicture = topic_model.visualize_topics_over_time(topics_over_time)
#     DynamicPicture.write_html("D:/python/Bertopic/K VS H/DynamicPicture"+str(item)+".html")

timestamps= data['priority date']
timestamps=timestamps.tolist()
topics_over_time = topic_model.topics_over_time(abstracts, timestamps,nr_bins=30,datetime_format="%Y-%m-%d")
DynamicPicture=topic_model.visualize_topics_over_time(topics_over_time)
DynamicPicture.write_html("D:/python/Bertopic/可视化结果/DynamicPictureF.html")