import numpy as np
import random
import typing

class Neuron:
    def __init__(self, layer: int, value = 0):
        self.value = value
        self.layer = layer

class InputNeuron(Neuron):
    def __init__(self):
        self.successors = []
        Neuron.__init__(self, layer=0)

class HiddenNeuron(Neuron):
    def __init__(self, layer: int):
        self.successors = []
        # predecessor (Neuron, edge weight)
        self.predecessors = []
        Neuron.__init__(self, layer=layer)

    def calc_value(self):
        weighted_sum = 0
        for predecessor in self.predecessors:
            weighted_sum += predecessor[0].value * predecessor[1]
        if weighted_sum > 0:
            self.value = 1
        elif weighted_sum < 0:
            self.value = -1
        else:
            self.value = 0

class OutputNeuron(Neuron):
    def __init__(self, layer: int):
        # predecessor (Neuron, edge weight)
        self.predecessors = []
        Neuron.__init__(self, layer=layer)

    def calc_value(self):
        weighted_sum = 0
        for predecessor in self.predecessors:
            weighted_sum += predecessor[0].value * predecessor[1]
        if weighted_sum > 0:
            self.value = 1
        elif weighted_sum < 0:
            self.value = -1
        elif weighted_sum == 0:
            self.value = 0

class Network:
    def __init__(self):
        self.fitness = 0
        self.hidden_layers = 0
        # input neurons (1 per pixel)
        self.neurons_in = [InputNeuron() for x in range(486)]
        # output neurons (left, right, jump)
        self.neurons_out = [OutputNeuron(self.hidden_layers + 1) for x in range(3)]
        # hidden neurons
        self.neurons_hidden = []
        # gaussian distribution for probabilities
        self.gaussian = self.gaussian_distribution()


    def update_fitness(self, points, time):
        """
        Berechnet und aktualisiert den Fitness-Wert des Netzwerks 
        basierend auf den Punkten (des 'Spielers') und der vergangenen Zeit.
        """
        self.fitness = points - 50 * time

    def evaluate(self, values):
        """
        Wertet das Netzwerk aus. 
        
        Argumente:
            values: eine Liste von 27x18 = 486 Werten, welche die aktuelle diskrete Spielsituation darstellen
                    die Werte haben folgende Bedeutung:
                     1 steht fuer begehbaren Block
                    -1 steht fuer einen Gegner
                     0 leerer Raum
        Rueckgabewert:
            Eine Liste [a, b, c] aus 3 Boolean, welche angeben:
                a, ob die Taste "nach Links" gedrueckt ist
                b, ob die Taste "nach Rechts" gedrueckt ist
                c, ob die Taste "springen" gedrueckt ist.
        """

        # assuming hidden neurons are sorted by layer (ascending)

        if self.fitness > -10:
            # passing block values to input neurons
            for i in range(486):
                self.neurons_in[i].value = values[i]
            
            # calculating values of hidden neurons
            for hidden_neuron in self.neurons_hidden:
                hidden_neuron.calc_value()
            
            # calculating values of output neurons
            for output_neuron in self.neurons_out:
                output_neuron.calc_value()

            left, right, jump = [False, False, False]
            if self.neurons_out[0].value > 0:
                left = True
            if self.neurons_out[1].value > 0:
                right = True
            if self.neurons_out[2].value > 0:
                jump = True
            
            return [left, right, jump]

        return [False, False, False]

    def gaussian_distribution(self):
        x_axis = np.linspace(0, 26, 27)
        y_axis = np.linspace(0, 17, 18)
        x_axis, y_axis = np.meshgrid(x_axis, y_axis)

        mean = np.array([10.0, 9.5])
        variance = np.array([[18.0, 0], [0, 27.0]])

        pos = np.empty(x_axis.shape + (2,))
        pos[:, :, 0] = x_axis
        pos[:, :, 1] = y_axis

        n = mean.shape[0]
        Sigma_det = np.linalg.det(variance)
        Sigma_inv = np.linalg.inv(variance)
        N = np.sqrt((2*np.pi)**n * Sigma_det)
        # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
        # way across all the input variables.
        fac = np.einsum('...k,kl,...l->...', pos-mean, Sigma_inv, pos-mean)

        gauss = np.exp(-fac / 2) / N

        return gauss.reshape(-1)

    def search_cycles(self, start, stop, cycle = False):
        if start == stop:
            cycle = True
        else:
            if start.layer > stop.layer:
                for predecessor in start.predecessors:
                    if stop == predecessor[0]:
                        cycle = True
                    else:
                        cycle = self.search_cycles(predecessor[0], stop)
                    if cycle:
                        break
        return cycle

    def update_layers(self, start):
        if type(start) != OutputNeuron:
            for successor in start.successors:
                if successor[0].layer <= start.layer:
                    successor[0].layer = start.layer +1
                    if successor[0].layer > self.hidden_layers:
                        self.hidden_layers = successor[0].layer
                        for output_neuron in self.neurons_out:
                            output_neuron.layer = self.hidden_layers +1
                    self.update_layers(successor[0])

    def mutation_add_edge(self):
        # draw random neuron (input or hidden)
        count = len(self.neurons_in)+len(self.neurons_hidden)
        start_node = None
        if np.random.choice([0,1], p=[len(self.neurons_in)/count, len(self.neurons_hidden)/count]) == 0:
            start_node = random.choices(self.neurons_in, weights=self.gaussian)[0]
        else:
            start_node = np.random.choice(self.neurons_hidden)
        
        stop_node = None
        cycle = True
        while cycle:
            stop_node = np.random.choice(self.neurons_out + self.neurons_hidden)
            cycle = self.search_cycles(start_node, stop_node)

        # while True:
        #     stop_node = np.random.choice(self.neurons_out + self.neurons_hidden)
        #     if stop_node.layer > start_node.layer:
        #         break
        
        weight = np.random.choice([-1,1])
        start_node.successors.append([stop_node, weight])
        stop_node.predecessors.append([start_node, weight])
        if start_node.layer >= stop_node.layer:
            stop_node.layer = start_node.layer +1
            if stop_node.layer > self.hidden_layers:
                self.hidden_layers = stop_node.layer
            self.update_layers(stop_node)
            self.neurons_hidden.sort(key=lambda neuron: neuron.layer)

    def mutation_add_node(self):
        # choose random edge by first choosing end node 
        edge_bool = False
        node_b = None
        while not edge_bool:
            node_b = np.random.choice(self.neurons_out + self.neurons_hidden)
            if len(node_b.predecessors) > 0:
                edge_bool = True
        edge = np.random.choice(len(node_b.predecessors))

        weight = node_b.predecessors[edge][1]
        node_a = node_b.predecessors[edge][0]
        node_c = HiddenNeuron(node_a.layer +1)
        if node_c.layer > self.hidden_layers: # -> node_b can only be OutputNeuron
            self.hidden_layers = node_c.layer
            for output_neuron in self.neurons_out:
                output_neuron.layer = self.hidden_layers +1
        else:
            node_b.layer += 1
            if node_b.layer > self.hidden_layers:
                self.hidden_layers = node_b.layer
            self.update_layers(node_b)
        self.neurons_hidden.append(node_c)

        node_b.predecessors[edge][0] = node_c
        node_c.successors.append([node_b, weight])
        node_c.predecessors.append([node_a, 1])
        node_a.successors.remove([node_b, weight])
        node_a.successors.append([node_c, 1])

        node_c.calc_value()
        node_b.calc_value()
        self.neurons_hidden.sort(key = lambda neuron: neuron.layer)
