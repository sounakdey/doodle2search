import numpy as np
import pickle
import gensim.downloader as api
from gensim.models import Word2Vec

def create_class_embeddings(vocab, dataset = 'sketchy'):

    google_300_corpus = api.load("word2vec-google-news-300")
    model = Word2Vec(google_300_corpus)
    w2v_class = []
    for i, voca in enumerate(vocab):
        print(voca)
        w2v_class.append(np.array(model[voca]))
    w2v_class = np.array(w2v_class)
    filename_semantic = './data/semantic_labels_'+ dataset + '.npy'
    filename_vocab = './data/vocab_' + dataset + '.pkl'
    np.save(filename_semantic, w2v_class)
    pickle.dump(vocab, open(filename_vocab, 'wb'))

    return w2v_class

# bin_file = '/media/sounak/4tbdisk/Datasets/GoogleNews-vectors-negative300.bin'
# model = gensim.models.word2vec.Word2Vec.load_word2vec_format(bin_file, binary=True)
# with open('/home/sounak/Documents/Datasets/SBIR/Sketchy/photo/class_labels.txt', 'r') as f:
#     vocab = f.read().splitlines()

