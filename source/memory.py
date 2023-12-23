from collections import defaultdict
import numpy as np
import random

from config import MEMORY_DIMENSION, N_NEIGHBOURS_IN_CONTEXT_EMBEDDING, SINGLE_WORD_EMBEDDING_DIMENSION, \
    NEIGHBOUR_SHARE_OF_INFO_IN_EMBEDDING, ALPHA


# define the memory class
class Memory:
    def __init__(self, memory_dimension=MEMORY_DIMENSION, cell_dimension=SINGLE_WORD_EMBEDDING_DIMENSION) -> None:
        self.number_of_cells = MEMORY_DIMENSION
        self.cell_dimension = SINGLE_WORD_EMBEDDING_DIMENSION
        self.memory = defaultdict(lambda: np.zeros(self.cell_dimension))

    @property
    def occupied_cells(self):
        return len(self.memory)

    @property
    def free_cells(self):
        return self.number_of_cells - len(self.memory)

    def p_of_keeping_infos(self, current_position) -> np.array:
        normalizing_factor = 0
        p_of_keeping_infos = np.zeros(self.occupied_cells)
        index = 0
        for key, value in self.memory.items():
            elapsed_words = current_position + 1 - key
            p_of_keeping_infos[index] = (elapsed_words / (
                        elapsed_words + 1)) ** ALPHA  # watch for current position values, should always start from one
            normalizing_factor += p_of_keeping_infos[index]
            index += 1
        p_of_keeping_infos = p_of_keeping_infos / normalizing_factor
        return p_of_keeping_infos

    def clear_all_memory(self):
        for key in self.memory.keys():
            del self.memory[key]

    def clear_one_element(self, current_position=0, random_position=False):
        if random_position:
            key_to_delete = random.choice(list(self.memory.keys()))
        else:
            key_to_delete = np.random.choice(list(self.memory.keys()), p=self.p_of_keeping_infos(current_position).tolist())
        del self.memory[key_to_delete]

    def add_element_in_position_n(self, n, word_embedding):

        if n in self.memory.keys():
            raise ValueError("The cell corresponding to word {n} is already occupied")
        if self.free_cells == 0:
            self.clear_one_element()
        self.memory[n] = word_embedding

    def add_element(self, word_embedding, current_position, random_position=False):
        if self.free_cells == 0:
            self.clear_one_element(current_position=current_position, random_position=random_position, )
        self.memory[current_position] = word_embedding # pay attention to current position, should always start from 0

    def __repr__(self):
        memory_infos = f""
        for key, value in self.__dict__.items():
            memory_infos += f"{key} : {value} \n"
        return memory_infos

    def __str__(self) -> str:
        return f"Memory with {self.number_of_cells} cells and {self.occupied_cells} occupied cells"


if __name__ == "__main__":
    memory = Memory()
    print(memory.memory)
    print(memory.free_cells)
    print(memory.occupied_cells)
    memory.add_element_in_position_n(1, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(memory.memory)
    print(memory.free_cells)
    print(memory.occupied_cells)
    memory.add_element_in_position_n(2, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(memory.memory)
    print(memory.free_cells)
    print(memory.occupied_cells)
    memory.add_element_in_position_n(3, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    print(memory.memory)
    print(memory.free_cells)
    print(memory.occupied_cells)

    print(memory.memory)
    print(memory.free_cells)
    print(memory.occupied_cells)

    print(memory.get_element(1))
    print(memory.get_element(2))
    print(memory.get_element(3))
