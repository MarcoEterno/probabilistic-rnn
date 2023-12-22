import numpy as np
from collections import defaultdict
import random

from config import N_NEIGHBOURS_IN_CONTEXT_EMBEDDING, SINGLE_WORD_EMBEDDING_DIMENSION, \
    NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING, ALPHA


class Text:
    def __init__(self, text: str = None) -> None:
        self.text = []
        if text:
            for word in text.split():
                self.text.append(word)
        self.embedded_text = []
        self.contextually_embedded_text = []
        self.vocabulary = defaultdict()  # creates the correspondence between words and embeddings
        self.embedding = lambda x: np.zeros(SINGLE_WORD_EMBEDDING_DIMENSION)

    def add_text(self, text: str):
        for word in text.split():
            self.text.append(word)

    def embed_vocabulary_one_hot(self):
        """
        Creates a one hot encoding for the words in the vocabulary with dimension 
        equal to the number of words in the vocabulary

        """
        if not self.vocabulary.keys:
            raise ValueError("For a one hot embedding to be initialized the vocabulary must not be empty")
        for index, word in enumerate(self.vocabulary.keys()):
            self.vocabulary[word] = np.zeros(len(self.vocabulary))
            self.vocabulary[word][index] = 1

    def generate_one_hot_vocabulary_for_text(self) -> None:
        """
        Creates a one hot encoding for the words in the text and saves it in the vocabulary attribute
        :return:
        """
        for index, word in enumerate(self.text):
            self.vocabulary[word] = np.zeros(len(self.text))
            self.vocabulary[word][index] = 1
            self.vocabulary[word] = self.vocabulary[word].tolist()

    def embed_text_for_current_vocabulary(self):
        for word in self.text:
            self.embedded_text.append(self.vocabulary[word])

    def add_word_to_vocabulary(self, word: str):
        if word not in self.vocabulary.keys():
            self.vocabulary[word] = len(self.vocabulary)
            self.inverted_vocabulary[len(self.vocabulary) - 1] = word

    def add_word_embedding(self, word_embedding: list):
        self.embedded_text.append(word_embedding)

    def add_word(self, word: str):
        self.text.append(word)

    def generate_n_hot_encoded_text_embedding(self, n) -> list:
        text_embeddings = []
        for i in range(n):
            new_embedding = np.zeros(SINGLE_WORD_EMBEDDING_DIMENSION)
            new_embedding[random.randint(0, SINGLE_WORD_EMBEDDING_DIMENSION - 1)] = 1
            text_embeddings.append(new_embedding)
        self.embedded_text.append(text_embeddings)
        return text_embeddings

    # NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING = 0.5 # Quantifies the fraction of the words embedding dedicated to the informations of the neighbouring words.

    def find_contextually_embedded_text(self):
        for i in range(len(self.embedded_text)):
            partial_contextual_embedding = (1 - NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING) * np.array(self.embedded_text[i])
            if i == 0:
                self.contextually_embedded_text.append(self.embedded_text[i])
                continue
            if i < N_NEIGHBOURS_IN_CONTEXT_EMBEDDING:
                # If there are not enough neighbours on the left of a word, each one will count more towards the contextual embedding
                renormalization_factor = NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING / i
                for j in range(0, i):
                    partial_contextual_embedding += np.array(self.embedded_text[j]) * renormalization_factor
            else:
                for j in range(i, i + N_NEIGHBOURS_IN_CONTEXT_EMBEDDING):
                    renormalization_factor = NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING / N_NEIGHBOURS_IN_CONTEXT_EMBEDDING
                    partial_contextual_embedding += np.array(self.embedded_text[j]) * renormalization_factor
            print(partial_contextual_embedding)
            self.contextually_embedded_text.append(partial_contextual_embedding.tolist())

    def __str__(self):
        return (f"Text: {self.text}\n"
                f"Embedded text: {self.embedded_text}\n"
                f"Contextually embedded text: {self.contextually_embedded_text}\n"
                f"Vocabulary: {self.vocabulary}\n")

    def __len__(self):
        return len(self.text)


if __name__ == "__main__":
    text = Text("Hi how are you today")

    text.generate_one_hot_vocabulary_for_text()
    text.embed_text_for_current_vocabulary()
    text.find_contextually_embedded_text()

    print(f"contextual embedding: {text.contextually_embedded_text}\n")
    print(f"embedding: {text.embedded_text}\n")
    print(f"vocabulary: {text.vocabulary}\n")
