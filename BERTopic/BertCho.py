import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
import warnings
import gensim
from gensim import corpora
from gensim.models.coherencemodel import CoherenceModel
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
def main():
    data = pd.read_excel("Result4.xlsx")  # content type
    abstracts = data['content_cutted']

    # Precompute embeddings
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = embedding_model.encode(abstracts, show_progress_bar=True)

    # Dimensionality reduction of embeddings
    umap_model = UMAP(n_neighbors=15, n_components=60, min_dist=0.01, metric='cosine', random_state=42)


    vectorizer_model = CountVectorizer(analyzer='word',
                                       min_df=1,  # minimum reqd occurences of a word
                                       stop_words='english',  # remove stop words
                                       lowercase=True,  # convert all words to lowercase
                                       token_pattern='[a-zA-Z0-9]{3,}',  # num chars > 3
                                       max_features=4000,  # max number of uniq words
                                       )


    Clusternumber=[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200,210,220,230,240,250,260,270,280,290,300]
    Clusternumber.reverse()
    print(Clusternumber)

    All=[]
    CoherenceScore = np.zeros((40, 40))
    MeanValue=np.zeros((40, 40))
    Variance=np.zeros((40, 40))

    for i in Clusternumber:
        # Cluster documents
        hdbscan_model = HDBSCAN(min_cluster_size=i, metric='euclidean', cluster_selection_method='eom',
                                prediction_data=True)
        # Create model
        topic_model = BERTopic(
            # Pipeline models
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            # representation_model=representation_model,
            # Hyperparameters
            top_n_words=20,
            # When training a BERTopic model and setting verbose=True, you may see various log information during the model training process.
            # verbose=True,
            # nr_topics=topicsize
        )
        topic_model.fit_transform(abstracts, embeddings)
        topics2 = topic_model.get_topic_info()['Topic'].values
        topicsize = len(topics2)+1
        if(topicsize>30):
            topicsize=30
        for j in range(2,topicsize):

            topic_model = BERTopic(
                # Pipeline models
                embedding_model=embedding_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer_model,
                # representation_model=representation_model,
                # Hyperparameters
                top_n_words=20,
          
                # verbose=True,
                nr_topics=j
            )
            topic_model.fit_transform(abstracts, embeddings)
            # Calculate the similarity mean and variance
            similarity_map = topic_model.visualize_heatmap()
            b = similarity_map.data[0].__getattribute__("z")
            np_matrix = np.array(b)
            # Calculate the mask of the lower triangular matrix, the diagonals are also excluded
            lower_triangle_mask = np.tril(np_matrix, -1)
            # Extract non-zero elements of the lower triangular part
            lower_triangle_elements = lower_triangle_mask[lower_triangle_mask != 0]
            # Calculate mean
            mean_value = np.mean(lower_triangle_elements)
            # Calculate variance
            variance_value = np.var(lower_triangle_elements)

            # Calculate the consistency score
            topics = topic_model.topics_
            topics2 = topic_model.get_topic_info()['Topic'].values
            documents = pd.DataFrame({"Document": abstracts,
                                      "ID": range(len(abstracts)),
                                      "Topic": topics})
            documents_per_topic = documents.groupby(['Topic'], as_index=False).agg({'Document': ' '.join})
            cleaned_docs = topic_model._preprocess_text(documents_per_topic.Document.values)
            vectorizer = topic_model.vectorizer_model
            analyzer = vectorizer.build_analyzer()
            words = vectorizer.get_feature_names_out()
            tokens = [analyzer(doc) for doc in cleaned_docs]
            dictionary = corpora.Dictionary(tokens)
            corpus = [dictionary.doc2bow(token) for token in tokens]
            topic_words = [[words for words, _ in topic_model.get_topic(topic)]
                           for topic in range(0,len(topics2)-1)]
            coherence_model = CoherenceModel(topics=topic_words,
                                             texts=tokens,
                                             corpus=corpus,
                                             dictionary=dictionary,
                                             coherence='c_v')
            coherence = coherence_model.get_coherence()
            topicnumber= len(topics2)


            print(str(topicnumber)+"BERTopic coherence score: {}".format(coherence)+"      clusterSize"+str(i))
            print(str(topicnumber) +"BERTopic mean_value: {}".format(mean_value) + "      clusterSize" + str(i))
            print(str(topicnumber) + "BERTopic variance_value: {}".format(variance_value) + "      clusterSize" + str(i))
            CoherenceScore[i//10][j]=coherence
            MeanValue[i//10][j]=mean_value
            Variance[i//10][j]=variance_value
    All.append(CoherenceScore)
    All.append(MeanValue)
    All.append(Variance)
    return All
if __name__ == '__main__':
    Result=main()