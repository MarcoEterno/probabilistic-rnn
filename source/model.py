from collections import defaultdict
import numpy as np
from matplotlib import pyplot as plt

from text import Text
from memory import Memory
from config import MEMORY_DIMENSION, N_NEIGHBOURS_IN_CONTEXT_EMBEDDING, SINGLE_WORD_EMBEDDING_DIMENSION, \
    NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING, ALPHA


class Model:
    def __init__(self, text: Text, memory: Memory) -> None:
        self.text = text
        self.memory = memory
        self.current_position = 0

    def move_one_word_forward(self):
        if self.current_position >= len(self.text.contextually_embedded_text) - 1:
            raise ValueError("The end of the text has been reached")
        self.current_position += 1
        self.memory.add_element(self.text.contextually_embedded_text[self.current_position])

    def calculate_information_presence_in_memory(self):
        information_presence = []
        # implementing the scalar product formula discussed in the notebook
        for word_embedding in self.text.embedded_text:
            info_about_word = 0
            for vector in self.memory.memory.values():
                info_about_word += np.dot(np.array(word_embedding), np.array(vector))
            information_presence.append(info_about_word)
        return information_presence

    def plot_information_presence_in_memory(self, information_presence, show_labels=False):
        for i in range(len(information_presence)):
            if show_labels:
                plt.scatter(i, information_presence[i], marker="o", label=model.text.text[i])
                plt.legend(loc="upper right")
            else:
                plt.scatter(i, information_presence[i], marker="o")
        plt.show()

    def __repr__(self):
        model_infos = f""
        for key, value in self.__dict__.items():
            model_infos += f"{key} : {value} \n"
        return model_infos


if __name__ == "__main__":
    text = Text("Hi, my name is Linus. I am a student at the University of Copenhagen. I am studying computer science."
                " I am currently working on a project about memory and information. This has the potential to "
                "revolutionize the way large language models work.")
    text.generate_one_hot_vocabulary_for_text()
    text.embed_text_for_current_vocabulary()
    text.find_contextually_embedded_text()
    print(text.contextually_embedded_text)

    memory = Memory(memory_dimension=5)

    model = Model(text, memory)

    for i in range(len(text) - 1):
        print("Current position: ", model.current_position)
        model.move_one_word_forward()
    print("Memory: ", model.memory.memory.__repr__())

    information_presence = model.calculate_information_presence_in_memory()
    model.plot_information_presence_in_memory(information_presence)
