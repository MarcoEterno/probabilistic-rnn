"""
This script generates a text file with 1000 unique words. it is used for debugging purposes.
"""

import random
import string

NUMBER_OF_WORDS = 10000

# Function to generate a word of random letters
def generate_word(length=5):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

# Set to hold the unique words
unique_words = set()

# Generate 1000 unique words
while len(unique_words) < NUMBER_OF_WORDS:
    word = generate_word()
    unique_words.add(word)

# Joining the words into a single string separated by a space
text = ' '.join(unique_words)

# Saving the text to a file
with open(f'{NUMBER_OF_WORDS}_non_repeating_words.txt', 'w') as f:
    f.write(text)