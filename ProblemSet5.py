# Lab 5 Report

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

%matplotlib inline
import dynworm as dw

## Exercise 1: Visualize brain connectomes

<img src="lab5_exercise1.png" width="1000">

# Load synaptic connectome and neuron classes

celegans_syn_conn_pd = pd.read_excel('connectome_syn.xlsx')
celegans_syn_conn_np = np.array(celegans_syn_conn_pd)

# Classes are ordered according to the neurons' order in synaptic connectome
neuron_classes = np.load('neuron_classes.npy') 

# Adjacency matrix of the first 10 neurons
print(celegans_syn_conn_np[:10, :10])

# Neuron classes of the first 10 neurons
print(neuron_classes[:10])

def vis_conn(syn_conn, neuron_classes):

    
    # YOUR CODE HERE
    # The function should output a 1 x 3 subplot
    # For each subplot, make sure to set plt.ylim(len(sub_adjacency_matrix), 0) so that first row starts with neuron 0
    # Add appropriate x,y labels for each subplot (e.g. sensory vs sensory, inter vs inter, etc)
    
    # Constants for neuron classes
    #Neurons consists of 3 classes. Sensory, Inter, Motor
    SENSORY = 'sensory'
    INTER = 'inter'
    MOTOR = 'motor'
    
    # Identify indices for each neuron class
    sensory_indices = [i for i, n in enumerate(neuron_classes) if n == SENSORY]
    inter_indices = [i for i, n in enumerate(neuron_classes) if n == INTER]
    motor_indices = [i for i, n in enumerate(neuron_classes) if n == MOTOR]
    
    # Extract sub-networks
    # seperating each class from the overall graph of neurons
    #by creating seperate matricies for each class. 
    #sensory_subnetwork = syn_conn[(sensory_indices, sensory_indices)]
    sensory_subnetwork = syn_conn[np.ix_(sensory_indices, sensory_indices)]
    inter_subnetwork = syn_conn[np.ix_(inter_indices, inter_indices)]
    motor_subnetwork = syn_conn[np.ix_(motor_indices, motor_indices)]
    # np.ix creates the submatrix
    # Plotting
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    
    for ax, matrix, title in zip(axs, 
                                 [sensory_subnetwork, inter_subnetwork, motor_subnetwork], 
                                 [SENSORY.capitalize(), INTER.capitalize(), MOTOR.capitalize()]):
        c = ax.pcolor(matrix, cmap='Greys', vmin=0, vmax=1)
        ax.set_title(f"{title} Neurons Sub-network")
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Neuron Index')
        ax.invert_yaxis()  # Invert y-axis
        
	
    plt.tight_layout()
    plt.show()



# Test your function here

vis_conn(syn_conn = celegans_syn_conn_np, neuron_classes = neuron_classes)

## Exercise 2: Locating the most connected vertices

<img src="lab5_exercise2.png" width="1000">

# Load the synaptic connectome and sample social network

syn_conn_pd = pd.read_excel('connectome_syn.xlsx')
syn_conn_np = np.array(syn_conn_pd)

social_network_sample_pd = pd.read_excel('social_network_sample.xlsx')
social_network_sample_np = np.array(social_network_sample_pd)

def find_hub_vertices(adj_mat, num_vertices):
    
    # Calculate in-degree and out-degree
    in_degrees = np.sum(adj_mat, axis=0)  # Sum columns for in-degree
    out_degrees = np.sum(adj_mat, axis=1)  # Sum rows for out-degree

    # Find indices of vertices with the highest in-degrees and out-degrees
    highest_in_degrees_indices = np.argsort(-in_degrees)[:num_vertices]
    highest_out_degrees_indices = np.argsort(-out_degrees)[:num_vertices]

    # Convert numpy arrays to lists for output
    indegree_list = highest_in_degrees_indices.tolist()
    outdegree_list = highest_out_degrees_indices.tolist()

    
    return indegree_list, outdegree_list


# Test your function with synaptic connectome

indegree_list_syn_conn, outdegree_list_syn_conn = find_hub_vertices(adj_mat = syn_conn_np, num_vertices = 10)

print(indegree_list_syn_conn)

print(outdegree_list_syn_conn)

# Test your function with sample social media network

indegree_list_SN, outdegree_list_SN = find_hub_vertices(adj_mat = social_network_sample_np, num_vertices = 5)

print(indegree_list_SN)

print(outdegree_list_SN)

## Exercise 3: Removing vertices from a graph

<img src="lab5_exercise3.png" width="1000">

# We will use the pre-existing directed graph sample earlier in the lab

directed_adj_mat_pd = pd.read_excel('directed_sample.xlsx')
directed_adj_mat_np = np.array(directed_adj_mat_pd)

def remove_vertices(adj_mat, vertices_2b_removed):
    # YOUR CODE HERE
    
    # Copy over matrix from original matrix to ensure original matrix is not affected
    adj_mat_new = np.copy(adj_mat)
    
    
    for vertex in vertices_2b_removed:
        
        # Look through row and column of vertex and reset them to 0, effectively removing any connection to vertex
        
        for i in range(0, len(adj_mat_new)):
            adj_mat_new[vertex][i] = 0
        for i in range(0, len(adj_mat_new[0])):
            adj_mat_new[i][vertex] = 0
    
    # Return the matrix with removed vertices
    return adj_mat_new


vertices_2b_removed_1 = [0, 5]        # Vertices to be removed set 1
vertices_2b_removed_2 = [1, 2, 6]     # Vertices to be removed set 2

# Test your function with set 1

directed_adj_mat_new_1 = remove_vertices(adj_mat = directed_adj_mat_np, vertices_2b_removed = vertices_2b_removed_1)

# Test your function with set 2

directed_adj_mat_new_2 = remove_vertices(adj_mat = directed_adj_mat_np, vertices_2b_removed = vertices_2b_removed_2)

### Original graph image for reference

<img src="directed_sample_graph.png" width="400">

# Using networkX, plot your directed graph with removed vertices according to vertices_2b_removed_1
# Use circular graph layout
# Label your edges according to their weights

directed_adj_mat_new_1_nx = nx.from_numpy_array(directed_adj_mat_new_1, create_using=nx.DiGraph())
pos=nx.circular_layout(directed_adj_mat_new_1_nx) # Establish that this network will be in a circular layout
# Draw the graph
nx.draw_networkx(directed_adj_mat_new_1_nx, pos, with_labels = True, node_size = 750, node_color='grey')
labels = nx.get_edge_attributes(directed_adj_mat_new_1_nx,'weight') # Create labels
nx.draw_networkx_edge_labels(directed_adj_mat_new_1_nx, pos, edge_labels=labels) # Implement labels
plt.axis('off')
plt.show()



# Using networkX, plot your directed graph with removed vertices according to vertices_2b_removed_2
# Use circular graph layout
# Label your edges according to their weights
directed_adj_mat_new_2_nx = nx.from_numpy_array(directed_adj_mat_new_2, create_using=nx.DiGraph())
pos=nx.circular_layout(directed_adj_mat_new_2_nx) # Establish that this network will be in a circular layout
# Draw the graph
nx.draw_networkx(directed_adj_mat_new_2_nx, pos, with_labels = True, node_size = 750, node_color='grey')
labels = nx.get_edge_attributes(directed_adj_mat_new_2_nx,'weight') # Create labels
nx.draw_networkx_edge_labels(directed_adj_mat_new_2_nx, pos, edge_labels=labels) # Implement labels
plt.axis('off')
plt.show()

## Exercise 4: Adding a new vertex to a graph

<img src="lab5_exercise4.png" width="1000">

# We will use the pre-existing directed graph sample earlier in the lab
# The graph has 7 vertices

directed_adj_mat_pd = pd.read_excel('directed_sample.xlsx')
directed_adj_mat_np = np.array(directed_adj_mat_pd)

def add_vertex(adj_mat, outgoing_edges, incoming_edges):
    
    # YOUR CODE HERE
    # The original directed graph has 7 vertices
    # The new vertex to be added can be regarded as 8th vertex of the graph
    # You can assume that each edge being added has weight of 1
    adj_mat_new = np.zeros((len(adj_mat) + 1, len(adj_mat[0]) + 1)).astype(int)
    
    # Copy over original matrix values into new matrix, leaving additional row and column as zeroes
    adj_mat_new[0:len(adj_mat), 0:len(adj_mat[0])] = adj_mat
    
    # Set every outgoing and incoming edge for the new row and column to 1
    for edge in outgoing_edges:
        adj_mat_new[len(adj_mat_new) - 1, edge] = 1
    for edge in incoming_edges:
        adj_mat_new[edge, len(adj_mat_new[0]) - 1] = 1
        
        
    return adj_mat_new
    
    

# Define outgoing and incoming edges for the new vertex to be added

outgoing_edges = [2, 3, 5]
incoming_edges = [3, 4, 6]

# Test your function with provided list of outgoing/incoming edges

directed_adj_mat_vertex_added = add_vertex(adj_mat = directed_adj_mat_np, 
                                  outgoing_edges = outgoing_edges, 
                                  incoming_edges = incoming_edges)

### Original graph image for reference

<img src="directed_sample_graph.png" width="400">

# Using networkX, plot your directed graph with added vertices according to outgoing_edges and incoming_edges
# Use circular graph layout
# Label your edges according to their weights

# YOUR CODE HERE
directed_adj_mat_vertex_added_nx = nx.from_numpy_array(directed_adj_mat_vertex_added, create_using=nx.DiGraph())
pos=nx.circular_layout(directed_adj_mat_vertex_added_nx) # Establish that this network will be in a circular layout
# Draw the graph
nx.draw_networkx(directed_adj_mat_vertex_added_nx, pos, with_labels = True, node_size = 750, node_color='grey')
labels = nx.get_edge_attributes(directed_adj_mat_vertex_added_nx,'weight') # Create labels
nx.draw_networkx_edge_labels(directed_adj_mat_vertex_added_nx, pos, edge_labels=labels) # Implement labels
plt.axis('off')
plt.show()

## Exercise 5: Re-wire neurons to restore behavior of C. elegans

<img src="lab5_exercise5.png" width="1000">

### Note: If you wish to use the included C. elegans simulation code in lab template folder outside of EE 241 (e.g. research purpose), please cite the following paper 
### Kim, J., Leahy, W., & Shlizerman, E. (2019). Neural interactome: Interactive simulation of a neuronal system. Frontiers in Computational Neuroscience, 13, 8. 

# Load synaptic connectome and neuron classes

damaged_syn_conn_pd = pd.read_excel('connectome_syn.xlsx')
damaged_syn_conn_np = np.array(damaged_syn_conn_pd)

## Motorneurons' activities during gentle tail touch (Damaged brain)

<img src="damaged_AVA_motor_activities.png" width="450">

## Simulated body movement during gentle tail touch (Damaged brain)

from ipywidgets import Video

Video.from_file("escape_response_damaged.mp4", width=500, height=500)

# Re-wiring instructions for AVAL and AVAR neurons

outgoing_AVAL_triples = np.load('AVAL_outgoing_triples.npy') # AVAL is the 47th vertex in the graph
incoming_AVAL_triples = np.load('AVAL_incoming_triples.npy') # AVAL is the 47th vertex in the graph

outgoing_AVAR_triples = np.load('AVAR_outgoing_triples.npy') # AVAR is the 55th vertex in the graph
incoming_AVAR_triples = np.load('AVAR_incoming_triples.npy') # AVAR is the 55th vertex in the graph

# Each row in the triple is ordered as [Source neuron index, Target neuron index, Synaptic weight]

rewiring_triples_AVAL = [outgoing_AVAL_triples, incoming_AVAL_triples]
rewiring_triples_AVAR = [outgoing_AVAR_triples, incoming_AVAR_triples]

def rewire_neurons(damaged_synaptic_adj_matrix, rewiring_instructions_AVAL, rewiring_instructions_AVAR):
    
    # YOUR CODE HERE
    # AVAL, AVAR neurons take the indices of 47, 55 respectively in the damaged_syn_conn_np
    AVAL_index = 47  # AVAL neuron index
    AVAR_index = 55  # AVAR neuron index

    # Repair connections for AVAL
    for triple in rewiring_instructions_AVAL[0]:  # Outgoing connections from AVAL
        source, target, weight = triple
        damaged_synaptic_adj_matrix[AVAL_index, target] = weight
    
    for triple in rewiring_instructions_AVAL[1]:  # Incoming connections to AVAL
        source, target, weight = triple
        damaged_synaptic_adj_matrix[source, AVAL_index] = weight

    # Repair connections for AVAR
    for triple in rewiring_instructions_AVAR[0]:  # Outgoing connections from AVAR
        source, target, weight = triple
        damaged_synaptic_adj_matrix[AVAR_index, target] = weight
    
    for triple in rewiring_instructions_AVAR[1]:  # Incoming connections to AVAR
        source, target, weight = triple
        damaged_synaptic_adj_matrix[source, AVAR_index] = weight

    # The matrix is now repaired, so we return it with a more appropriate name
    repaired_synaptic_adj_matrix = damaged_synaptic_adj_matrix
    return repaired_synaptic_adj_matrix


repaired_synaptic_adj_matrix = rewire_neurons(damaged_synaptic_adj_matrix = damaged_syn_conn_np, 
                                              rewiring_instructions_AVAL = rewiring_triples_AVAL, 
                                              rewiring_instructions_AVAR = rewiring_triples_AVAR)

# Test your repaired connectome 

dw.network_sim.test_brain_repair(repaired_synaptic_adj_matrix) 

# If successfully repaired, function will output 
# 1) Motorneurons activity 
# 2) Simulated body movement video with repaired brain

