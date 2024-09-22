import gensim
from gensim import corpora
from gensim.models.ldamodel import LdaModel
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import matplotlib.pyplot as plt


def plot_grid_search(cv_results, grid_param_1):
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results
    # scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))
    # Plot Grid search scores
    _, ax = plt.subplots(1,1)
    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    # for idx, val in enumerate(grid_param_2):
    #     ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))
    ax.plot(grid_param_1, scores_mean, '-o')
    ax.set_xlabel("Topic number", fontsize=16)
    ax.set_ylabel('Coherence', fontsize=16)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    plt.show()
def main():
    # Assume that `documents` is your original document list, and each document is a preprocessed and segmented word list.
    # documents = [['word1', 'word2', ...], ['word1', 'word2', ...], ...]
    data=pd.read_excel("D:/python/Projects/LDA/data/delet stopwords1.xlsx")#content type
    # First, prepare the text data
    texts = [text.split() for text in data.content_cutted]
    #Create dictionary
    dictionary = corpora.Dictionary(texts)
    #Create corpus
    corpus = [dictionary.doc2bow(doc) for doc in texts]
     # update_every: This parameter controls the training frequency of the model. update_every=1 means that the model will be updated after each batch of documents is processed, which is a setting for online learning or small batch learning. If set to 0, the model will perform a full update after all documents have been processed, which is suitable for batch learning.
     #
     # chunksize: This parameter controls the number of documents used each time the model is trained. Chunksize affects the memory consumption and training speed of model training. A larger chunksize can increase the training speed, but it will also increase memory usage.
     #
     # passes: This parameter specifies how many times the entire corpus should be passed. Multiple passes can improve model stability and performance, but also increase training time.
     #
     # alpha: This is a parameter related to the prior of the document-topic distribution. alpha='auto' allows the model to automatically learn how sparse the topic distribution is for each document. Lower alpha values tend to result in fewer topics being assigned to each document, while higher alpha values allow documents to contain more topics.
     #
     # per_word_topics: If set to True, the model will additionally calculate and save the most likely topic distribution for each word at the end of training. This is useful for examining relationships between words and topics, but adds overhead to model training and model size after saving.
     # Train LDA model
    CoherenceScore = []
    params = []
    for i in range(2, 26):
        params.append(i)
        lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=i, random_state=100,
                             passes=50, alpha=0.5, eta=0.1)
        # Calculate and print the consistency score of the model
        coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_lda = coherence_model_lda.get_coherence()
        CoherenceScore.append(coherence_lda)
        print("Topicnumber:"+str(i)+",coherenceScore="+str(coherence_lda))
    plot_grid_search(CoherenceScore, params)
    plt.show()


if __name__ == '__main__':
    main()
