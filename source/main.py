import random
from collections import defaultdict
import numpy as np

from config import MEMORY_DIMENSION, N_NEIGHBOURS_IN_CONTEXT_EMBEDDING, SINGLE_WORD_EMBEDDING_DIMENSION, \
    NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING, ALPHA
from text import Text
from memory import Memory
from model import Model
from utils import get_paul_graham_essay, get_non_repeating_text

if __name__ == "__main__":
    long_text = get_non_repeating_text(2000)
    print("loaded text of lenght: ", len(long_text.split()), " words")
    text = Text(long_text)
    print("generating vocabulary")
    text.generate_one_hot_vocabulary_for_text()
    print("embedding vocabulary")
    text.embed_text_for_current_vocabulary()
    print("crreating contextually embedded text")
    text.find_contextually_embedded_text()
    print("creating memory")
    memory = Memory(memory_dimension=MEMORY_DIMENSION)
    print("creating model")
    model = Model(text, memory)

    print("starting simulation")
    for i in range(len(text) - 1):
        print("Current position: ", model.current_position)
        model.move_one_word_forward()

    information_presence = model.calculate_information_presence_in_memory(one_hot=True)
    model.plot_information_presence_in_memory(information_presence)

