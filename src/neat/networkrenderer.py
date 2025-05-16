import pygame

TILESIZE = 10

def drawNode(x, y, surface):
    # draw Neuron at (x,y)
    surface.fill((255, 255, 255), (x, y, TILESIZE, TILESIZE))
    pygame.draw.rect(surface, (0, 0, 0), (x, y, TILESIZE, TILESIZE), 1)

def drawEdge(xS, yS, xE, yE, color, surface):
    # draw Line from (xS,yS) to (xE,yE) in color c
    pygame.draw.line(surface, color, (xS+5, yS+5), (xE+5, yE+5), 1)

def render_network(surface, network, values):
    """
    Zeichnet die Minimap und den Netzwerkgraphen 
    
    Argumente:
        surface: ein pygame.Surface der Groesse 750 x 180 Pixel.
                 Darauf soll der Graph und die Minimap gezeichnet werden.
        network: das eigen implementierte Netzwerk (in network.py), dessen Graph gezeichnet werden soll.
        values: eine Liste von 27x18 = 486 Werten, welche die aktuelle diskrete Spielsituation darstellen
                die Werte haben folgende Bedeutung:
                 1 steht fuer begehbaren Block
                -1 steht fuer einen Gegner
                 0 leerer Raum
                Die Spielfigur befindet sich immer ca. bei der Position (10, 9) und (10, 10).
    """
    colors = {1: (255, 255, 255), -1: (255, 0, 0), 2: (0, 255, 0), 0: (255, 0, 0)}
    # draw slightly gray background for the minimap
    pygame.draw.rect(surface, (128, 128, 128, 128), (0, 0, 27 * TILESIZE, 18 * TILESIZE))
    # draw minimap
    for y in range(18):
        for x in range(27):
            if values[y * 27 + x] != 0:
                color = colors[values[y * 27 + x]]
                surface.fill(color, (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE))
                pygame.draw.rect(surface, (0, 0, 0), (TILESIZE * x, TILESIZE * y, TILESIZE, TILESIZE), 1)

    # draw neuronal network
    
    # draw hidden neurons
    if network.hidden_layers != 0:
        # arrange hidden neurons in matrix in respect to their layer
        neuron_mat = [[] for i in range(network.hidden_layers)]
        for i in range(len(network.neurons_hidden)):
            neuron_mat[network.neurons_hidden[i].layer -1].append(network.neurons_hidden[i])
        
        distance_layers = (740-280)/network.hidden_layers
        offset_x = distance_layers//2
        for layer in range(len(neuron_mat)):
            distance_nodes = 170
            if len(neuron_mat[layer]):
                distance_nodes = 170/len(neuron_mat[layer])
            offset_y = distance_nodes//2
            for neuron in range(len(neuron_mat[layer])):
                x_stop = 280 + layer * distance_layers + offset_x
                y_stop = neuron * distance_nodes + offset_y
                drawNode(x_stop, y_stop, surface)
                for predecessor in neuron_mat[layer][neuron].predecessors:
                    x_start = None
                    y_start = None
                    if predecessor[0].layer == 0:
                        index_start_node = network.neurons_in.index(predecessor[0])
                        x_start = (index_start_node % 27) * TILESIZE
                        y_start = (index_start_node // 27) * TILESIZE
                    else:
                        x_start = 280 + (predecessor[0].layer -1) * distance_layers + offset_x
                        distance_nodes_start = 170/len(neuron_mat[predecessor[0].layer-1])
                        y_start = neuron_mat[predecessor[0].layer-1].index(predecessor[0]) * distance_nodes_start + distance_nodes_start//2
                    drawEdge(x_start, y_start, x_stop, y_stop, colors[predecessor[1]+1], surface)

    # draw output neurons 
    distance_nodes = 170/len(network.neurons_out)
    offset_y = distance_nodes//2
    for neuron in range(len(network.neurons_out)):
        x_stop = 740
        y_stop = neuron * distance_nodes + offset_y
        drawNode(x_stop, y_stop, surface)
        for predecessor in network.neurons_out[neuron].predecessors:
            x_start = None
            y_start = None
            if predecessor[0].layer == 0:
                index_start_node = network.neurons_in.index(predecessor[0])
                x_start = (index_start_node % 27) * TILESIZE
                y_start = (index_start_node // 27) * TILESIZE
            else:
                x_start = 280 + (predecessor[0].layer -1) * distance_layers + offset_x
                distance_nodes_start = 170/len(neuron_mat[predecessor[0].layer-1])
                y_start = neuron_mat[predecessor[0].layer-1].index(predecessor[0]) * distance_nodes_start + distance_nodes_start//2
            drawEdge(x_start, y_start, x_stop, y_stop, colors[predecessor[1]+1], surface)
    