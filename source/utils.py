import os
import data


def get_paul_graham_essay():
    with open(os.path.join(data.__path__[0], "paul_graham_essay.txt"), "r") as f:
        return f.read()

def get_non_repeating_text(number_of_words=1000):
    if os.path.exists(os.path.join(data.__path__[0], f"{number_of_words}_non_repeating_words.txt")):
        with open(os.path.join(data.__path__[0], f"{number_of_words}_non_repeating_words.txt"), "r") as f:
            return f.read()
    else:
        text = ""
        for i in range(number_of_words):
            text += str(i) + " "
        with open(os.path.join(data.__path__[0], f"{number_of_words}_non_repeating_words.txt"), "w") as f:
            f.write(text)
        return text