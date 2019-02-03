import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Main function for user to choose parameters
def main():
    maximal_graph_flag = True

    if maximal_graph_flag:
        # Create maximal graph
        degree = 4
        levels = 4
        add_repeats = True
        graph, data = make_maximal_graph(degree, levels, add_repeats)
    else:
        num_nodes = 100
        graph = nx.random_tree(num_nodes)

    # Get graph information
    df, output_degree = get_graph_information(graph)

    # Plot graph with major nodes highlighted and with labels
    colour_values = []
    for node in df.index:
        if df.loc[node]['IsMajor']:
            colour_values.append('red')
        # elif df.loc[node]['IsTypeB']:
        #     colour_values.append('yellow')
        # elif df.loc[node]['IsTypeD']:
        #     colour_values.append('green')
        # elif df.loc[node]['IsTypeE']:
        #     colour_values.append('orange')
        else:
            colour_values.append('blue')

    if maximal_graph_flag:
        labels = dict(zip(data.index, data.Label.astype(int)))
    else:
        labels = dict(zip(df.index, df.index.astype(int)))
    pos = hierarchy_pos(graph, 0)
    nx.draw(graph, pos=pos, with_labels=True, labels=labels, node_color=colour_values)
    plt.show()
    print('nothing')


def make_maximal_graph(degree, levels, add_repeats=True):
    graph, data = initialise_maximal_graph(degree)
    for levels in range(2, levels+1):
        graph, data = add_level(graph, degree, data, add_repeats)
    return graph, data


def initialise_maximal_graph(degree):
    graph = nx.Graph()
    data = pd.DataFrame(
        {
            'Index': range(0, degree+1),
            'Label': [0] + list(range(2, degree+2)),
            'Parent': [np.nan] + [0]*degree,
            'Grandparent': [np.nan]*(degree+1),
            'Level': [0]+[1]*degree,
            'Triplet': [[np.nan]*3]*(degree+1)
         })
    data = data.set_index('Index')

    nodes = range(0, degree + 1)
    edges = [(0, i) for i in nodes[1:degree + 1]]
    graph.add_nodes_from(nodes)
    graph.add_edges_from(edges)
    return graph, data


# Function to add levels to a graph
def add_level(graph, degree, data, add_repeats=True):
    max_level = max(data['Level'])
    nodes_at_level = data.loc[data['Level'] == max_level].index.values.tolist()
    new_level = max_level + 1
    for parent_index in nodes_at_level:
        max_index = int(max(data.index))
        grandparent_index = int(data.loc[parent_index]['Parent'])
        parent_label = int(data.loc[parent_index]['Label'])
        grandparent_label = int(data.loc[grandparent_index]['Label'])

        all_labels = list(range(0, degree+2))
        parent_exclude = list(range(parent_label-1, parent_label+2))

        if not add_repeats:
            # Check old triplets
            triplets_old = [[i[0], i[1]] for i in data['Triplet']]
            # Get indices for which the grandparent and parent labels matches current
            index_gp = [i for i, j in enumerate(triplets_old) if j == [grandparent_label, parent_label]]
            # Add the labels corresponding to those indices
            labels_remove = [i[2] for i in data.loc[index_gp]['Triplet']]
        else:
            labels_remove = []

        new_labels = list(set(all_labels) - set(parent_exclude + [grandparent_label] + labels_remove))
        num_new_labels = len(new_labels)
        new_index = list(range(max_index + 1, max_index + num_new_labels + 1))

        graph.add_nodes_from(new_index)
        edges = [(parent_index, i) for i in new_index]
        graph.add_edges_from(edges)

        # Add additional information to DataFrame
        data_new = pd.DataFrame(
            {
                'Index': new_index,
                'Label': new_labels,
                'Parent': [parent_index]*num_new_labels,
                'Grandparent': [grandparent_index]*num_new_labels,
                'Level': [new_level]*num_new_labels})
        data_new = data_new.set_index('Index')
        data_new['Triplet'] = pd.Series([[data.loc[data_new.loc[index]['Grandparent']]['Label'], data.loc[data_new.loc[index]['Parent']]['Label'], data_new.loc[index]['Label']] for index in data_new.index], index=data_new.index)

        data = data.append(data_new)

    return graph, data


def neighborhood(graph, node, n):
    path_lengths = nx.single_source_dijkstra_path_length(graph, node)
    return [node for node, length in path_lengths.items() if length == n]


def get_graph_information(graph):
    # Get list of nodes
    nodes = graph.nodes()

    # Find distance 1 nodes, degree, is major
    distance_1_nodes = [neighborhood(graph, node, 1) for node in nodes]
    num_distance_1_nodes = [len(nodes) for nodes in distance_1_nodes]
    degree = max(num_distance_1_nodes)
    is_major = [num_nodes == degree for num_nodes in num_distance_1_nodes]
    distance_1_majors = [[node for node in nodes if is_major[node]] for nodes in distance_1_nodes]
    num_distance_1_majors = [len(nodes) for nodes in distance_1_majors]

    # Find distance 2 nodes
    distance_2_nodes = [neighborhood(graph, node, 2) for node in nodes]
    num_distance_2_nodes = [len(nodes) for nodes in distance_2_nodes]
    distance_2_majors = [[node for node in nodes if is_major[node]] for nodes in distance_2_nodes]
    num_distance_2_majors = [len(nodes) for nodes in distance_2_majors]

    # Find distance 3 nodes
    distance_3_nodes = [neighborhood(graph, node, 3) for node in nodes]
    num_distance_3_nodes = [len(nodes) for nodes in distance_3_nodes]
    distance_3_majors = [[node for node in nodes if is_major[node]] for nodes in distance_3_nodes]
    num_distance_3_majors = [len(nodes) for nodes in distance_3_majors]

    # Find type B nodes
    is_type_B = [num_distance_2_majors[node] == 2*degree-4 and not is_major[node] for node in nodes]
    distance_1_type_B = [[node for node in nodes if is_type_B[node]] for nodes in distance_1_nodes]
    num_distance_1_type_B = [len(nodes) for nodes in distance_1_type_B]
    distance_2_type_B = [[node for node in nodes if is_type_B[node]] for nodes in distance_2_nodes]
    num_distance_2_type_B = [len(nodes) for nodes in distance_2_type_B]
    distance_3_type_B = [[node for node in nodes if is_type_B[node]] for nodes in distance_3_nodes]
    num_distance_3_type_B = [len(nodes) for nodes in distance_3_type_B]

    # Find type D nodes
    is_type_D = [num_distance_2_majors[node] == 2*degree-5 and not is_major[node] for node in nodes]
    distance_1_type_D = [[node for node in nodes if is_type_D[node]] for nodes in distance_1_nodes]
    num_distance_1_type_D = [len(nodes) for nodes in distance_1_type_D]
    distance_2_type_D = [[node for node in nodes if is_type_D[node]] for nodes in distance_2_nodes]
    num_distance_2_type_D = [len(nodes) for nodes in distance_2_type_D]
    distance_3_type_D = [[node for node in nodes if is_type_D[node]] for nodes in distance_3_nodes]
    num_distance_3_type_D = [len(nodes) for nodes in distance_3_type_D]

    # Find type E nodes
    is_type_E = [num_distance_2_majors[node] == 2*degree-6 and not is_major[node] for node in nodes]
    distance_1_type_E = [[node for node in nodes if is_type_E[node]] for nodes in distance_1_nodes]
    num_distance_1_type_E = [len(nodes) for nodes in distance_1_type_E]
    distance_2_type_E = [[node for node in nodes if is_type_E[node]] for nodes in distance_2_nodes]
    num_distance_2_type_E = [len(nodes) for nodes in distance_2_type_E]
    distance_3_type_E = [[node for node in nodes if is_type_E[node]] for nodes in distance_3_nodes]
    num_distance_3_type_E = [len(nodes) for nodes in distance_3_type_E]

    # Store in DataFrame
    df = pd.DataFrame(
        {
            'Index': nodes,
            'IsMajor': is_major,
            'IsTypeB': is_type_B,
            'IsTypeD': is_type_D,
            'IsTypeE': is_type_E,
            'Distance1Nodes': distance_1_nodes,
            'NumDistance1Nodes': num_distance_1_nodes,
            'Distance1Majors': distance_1_majors,
            'NumDistance1Majors': num_distance_1_majors,
            'Distance1TypeB': distance_1_type_B,
            'NumDistance1TypeB': num_distance_1_type_B,
            'Distance1TypeD': distance_1_type_D,
            'NumDistance1TypeD': num_distance_1_type_D,
            'Distance1TypeE': distance_1_type_E,
            'NumDistance1TypeE': num_distance_1_type_E,
            'Distance2Nodes': distance_2_nodes,
            'NumDistance2Nodes': num_distance_2_nodes,
            'Distance2Majors': distance_2_majors,
            'NumDistance2Majors': num_distance_2_majors,
            'Distance2TypeB': distance_2_type_B,
            'NumDistance2TypeB': num_distance_2_type_B,
            'Distance2TypeD': distance_2_type_D,
            'NumDistance2TypeD': num_distance_2_type_D,
            'Distance2TypeE': distance_2_type_E,
            'NumDistance2TypeE': num_distance_2_type_E,
            'Distance3Nodes': distance_3_nodes,
            'NumDistance3Nodes': num_distance_3_nodes,
            'Distance3Majors': distance_3_majors,
            'NumDistance3Majors': num_distance_3_majors,
            'Distance3TypeB': distance_3_type_B,
            'NumDistance3TypeB': num_distance_3_type_B,
            'Distance3TypeD': distance_3_type_D,
            'NumDistance3TypeD': num_distance_3_type_D,
            'Distance3TypeE': distance_3_type_E,
            'NumDistance3TypeE': num_distance_3_type_E,
        })
    df = df.set_index('Index')
    return df, degree


# Makes the positions of the nodes look pretty
def hierarchy_pos(graph, root, width=1., vertical_gap=0.2, vertical_loc=0, x_center=0.5,
                  pos=None, parent=None):
    """If there is a cycle that is reachable from root, then this will see infinite recursion.
       graph: the graph
       root: the root node of current branch
       width: horizontal space allocated for this branch - avoids overlap with other branches
       vertical_gap: gap between levels of hierarchy
       vertical_loc: vertical location of root
       x_center: horizontal location of root
       pos: a dict saying where all nodes go if they have been assigned
       parent: parent of this branch."""

    if pos is None:
        pos = {root: (x_center, vertical_loc)}
    else:
        pos[root] = (x_center, vertical_loc)
    neighbors = list(graph.neighbors(root))
    if parent is not None:   # this should be removed for directed graphs.
        neighbors.remove(parent)  # if directed, then parent not in neighbors.
    if len(neighbors) != 0:
        dx = width/len(neighbors)
        next_x = x_center - width / 2 - dx / 2
        for neighbor in neighbors:
            next_x += dx
            pos = hierarchy_pos(graph, neighbor, width=dx, vertical_gap=vertical_gap,
                                vertical_loc=vertical_loc - vertical_gap, x_center=next_x, pos=pos,
                                parent=root)
    return pos


if __name__ == '__main__':
    main()
