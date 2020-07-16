import re
import numpy as np
import torch


class Preprocess:
    def __init__(self, config):
        self.config = config

    def preprocess(self,path,phase):
        stop_words = ['a', 'about', 'above', 'after', 'again', 'against', 'all', 'am', 'an', 'and', 'any', 'are', "aren't", 'as', 'at', 'be', 'because', 'been', 'before', 'being', 'below', 'between', 'both', 'but', 'by', "can't", 'cannot', 'could', "couldn't", 'did', "didn't", 'do', 'does', "doesn't", 'doing', "don't", 'down', 'during', 'each', 'few', 'for', 'from', 'further', 'had', "hadn't", 'has', "hasn't", 'have', "haven't", 'having', 'he', "he'd", "he'll", "he's", 'her', 'here', "here's", 'hers', 'herself', 'him', 'himself', 'his', 'how', "how's", 'i', "i'd", "i'll", "i'm", "i've", 'if', 'in', 'into', 'is', "isn't", 'it', "it's", 'its', 'itself', "let's", 'me', 'more', 'most', "mustn't", 'my', 'myself', 'no', 'nor', 'not', 'of', 'off', 'on', 'once', 'only', 'or', 'other', 'ought', 'our', 'ours', 'ourselves', 'out', 'over', 'own', 'same', "shan't", 'she', "she'd", "she'll", "she's", 'should', "shouldn't", 'so', 'some', 'such', 'than', 'that', "that's", 'the', 'their', 'theirs', 'them', 'themselves', 'then', 'there', "there's", 'these', 'they', "they'd", "they'll", "they're", "they've", 'this', 'those', 'through', 'to', 'too', 'under', 'until', 'up', 'very', 'was', "wasn't", 'we', "we'd", "we'll", "we're", "we've", 'were', "weren't", 'what', "what's", 'when', "when's", 'where', "where's", 'which', 'while', 'who', "who's", 'whom', 'why', "why's", 'with', "won't", 'would', "wouldn't", 'you', "you'd", "you'll", "you're", "you've", 'your', 'yours', 'yourself', 'yourselves']


        stop_words.remove("what")
        stop_words.remove("how")
        stop_words.remove("who")
        stop_words.remove("where")
        stop_words.remove("which")
        stop_words.remove("why")
        stop_words.remove("when")
        punctuation = '!,.;:?"\'`'

        
        input = open(path, 'r')


        labels = list()
        sentences = list()

        for line in input.readlines():
            labels.append(line.split()[0])
            sentence = line.lower().split()[1:]
            for word in line.lower().split()[1:]:
                if word in stop_words:
                    sentence.remove(word) # remove stop words
            sentences.append(re.sub(r'[{}]+'.format(punctuation), '', ' '.join(sentence))) # remove punctuation
            # sentences.append(' '.join(sentence))

        # About generating vocabulary
        # If it is training phase, then generate and save it
        if phase=="train":
            words_count_dict = {}
            
            for sentence in sentences:
                for word in sentence.split():
                    if word in words_count_dict:
                        words_count_dict[word] += 1
                    else:
                        words_count_dict[word] = 1
            vocabulary = list()
            wcd_sorted = list(words_count_dict.items())
            wcd_sorted.sort(key=lambda x:x[1],reverse=True)

            for word_count in wcd_sorted:
                if word_count[1]>int(self.config['min_words']):
                    vocabulary.append(word_count[0]) # form vocabulary in the order of word count
        # If it is testing phase, then just load the vocabulary
        if phase=="test":
            vocabulary = self.load("../data/vocabulary.bin")

        # About extracting the embeddings corresponding to the vocabulary from pretrained embeddings
        # If it is the training phase, do the extraction
        if phase=="train":
            # Get all of the embeddings
            emb = dict()
            with open(self.config['path_pre_emb'],'r') as f:
                for line in f.readlines():
                    line_list = line.split()
                    emb[line_list[0]] = np.array([float(val) for val in line_list[1:]])
            voca_embs = list()
            # Extract embeddings
            for word in vocabulary:
                try:
                    # If the embedding of the word exists in pretrained set, then get it
                    voca_embs.append(emb[word])
                except KeyError:
                    # If not, place the average of the pretrained embedding for words that are in the vocabulary but not found in pretrained set
                    voca_embs.append(emb['#UNK#'])
            # Place the average of the pretrained embedding for words that are in the sentence but not included in vocabulary
            voca_embs.append(emb['#UNK#'])
        # If it is the testing phase, load the extracted embeddings
        if phase=="test":
            voca_embs = self.load("../data/voca_embs.bin")

        # Generate sentence representations using vocabulary
        sens_rep = list()
        for sentence in sentences:
            sen_rep = list()
            for word in sentence.split():
                if word in vocabulary:
                    sen_rep.append(vocabulary.index(word))
                else:
                    # words that are in the sentence but not included in vocabulary
                    sen_rep.append(len(voca_embs)-1)
            sens_rep.append(torch.tensor(sen_rep))

        # About convert label to numeric representation
        # If it is the training phase, then generate the label to index conversion dictionary
        if phase=="train":
            labels_index = dict()
            count=0
            for label in labels:
                if label not in labels_index:
                    labels_index[label] = count
                    count += 1
        # If it is the testing phase, load it other than generate another to keep the labels consistent
        if phase=="test":
            labels_index = self.load("../data/labels_index.bin")
        # Generate labels' representations according to labels' index
        labels_rep = list()
        for label in labels:
            labels_rep.append(labels_index[label])


        # Save for further use
        self.save(labels,'../data/labels.bin')
        self.save(sentences,'../data/sentences.bin')
        self.save(vocabulary,'../data/vocabulary.bin')
        self.save(voca_embs,'../data/voca_embs.bin')
        self.save(sens_rep,'../data/sens_rep.bin')
        self.save(labels_index,'../data/labels_index.bin')
        self.save(labels_rep,'../data/labels_rep.bin')
        

        
    def load_preprocessed(self):
        return self.load('../data/labels.bin'),self.load('../data/sentences.bin'),self.load('../data/vocabulary.bin'),self.load('../data/voca_embs.bin'),self.load('../data/sens_rep.bin'),self.load('../data/labels_index.bin'),self.load('../data/labels_rep.bin')

    def save(self,l,f_name):
        torch.save(l,f_name)

    def load(self,f_name):
        return torch.load(f_name)