import copy
from datetime import datetime
from neat.network import Network
import pickle
import random

class Population():
    def __init__(self, seed, size):
        """
        Erstellt eine neue Population mit der Groesse 'size' und wird zuerst fuer den uebergebenen seed trainiert.
        """
        self.seed = seed
        # Das Attribut generation_count wird von Gadakeco automatisch inkrementiert. 
        self.generation_count = 1

        # eindeutiger name name des Netzwerks (noch zu implementieren)
        self.name = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

        self.current_generation = [Network() for i in range(size)]
        for network in self.current_generation:
            network.mutation_add_edge()

    @staticmethod
    def load_from_file(filename):
        """
        Laedt die komplette Population von der Datei mit dem Pfad filename.
        """
        return pickle.load(open(filename, "rb"))

    def save_to_file(self, filename):
        """
        Speichert die komplette Population in die Datei mit dem Pfad filename.
        """
        pickle.dump(self, open(filename, "wb"))

    def create_next_generation(self):
        """
        Erstellt die naechste Generation.
        """
        gen_count = len(self.current_generation)
        self.current_generation.sort(key=lambda network: network.fitness, reverse=True)

        for i in range(gen_count//10):
            for j in range(1, gen_count//10):
                self.current_generation[i+j*(gen_count//10)] = copy.deepcopy(self.current_generation[i])
        for i in range(gen_count//10, (int) (gen_count * 0.82)):
            self.current_generation[i].mutation_add_edge()
        for i in range((int) (gen_count * 0.82), gen_count):
            self.current_generation[i].mutation_add_node()
