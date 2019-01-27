import pandas as pd
import numpy as np
import networkx as nx
from itertools import chain

vertex_index_name = 'vertex_index'


def list_validator(input_data):
    output = input_data
    if not isinstance(output, list):
        output = [output]
    return output


def make_vertex_index(input_list=None):
    if input_list is None:
        input_list = list()
    return pd.Index(input_list, name=vertex_index_name, dtype='int64')


class Graph:

    _vdf = None
    _edges = None
    _nx_graph = None

    def initialise(self):
        self._vdf = pd.DataFrame(index=make_vertex_index())
        self._edges = set()
        self._nx_graph = nx.Graph()

    def __init__(self, vertices=None, n_vertices=None, vertex_attributes=None, edges=None, nx_graph=None):
        self.initialise()
        if nx_graph is not None:
            self.from_nx_graph(nx_graph=nx_graph)
        else:
            self.add_vertices(vertices=vertices, n_vertices=n_vertices)
            self.add_vertex_attributes(vertex_attributes=vertex_attributes)
            self.add_edges(edges=edges)

    @property
    def nx_graph(self):
        return self._nx_graph

    @property
    def vdf(self):
        return self._vdf

    @property
    def edf(self):
        vertices = self.vertices
        edf = pd.DataFrame(0, index=vertices, columns=vertices)
        edges = [(edge[0], edge[1], 1) for edge in self.edges]
        temp_edf = pd.DataFrame(edges, columns=['vertex_1', 'vertex_2', 'edge'])\
            .pivot('vertex_1', 'vertex_2', 'edge')\
            .fillna(0)\
            .astype('int64')
        edf.update(temp_edf)
        return edf

    @property
    def vertices(self):
        return self.vdf.index.tolist()

    @property
    def edges(self):
        return self._edges

    def add_vertices(self, vertices=None, n_vertices=None):
        if n_vertices is not None:
            start = int(np.nanmax([self.vdf.index.max(), -1]))
            vertices = list(range(start+1, start + n_vertices + 1))

        if vertices is not None:
            vertices = list_validator(vertices)
            self._nx_graph.add_nodes_from(vertices)
            new_index = pd.Index(vertices)
            vertex_index = make_vertex_index(new_index[~new_index.isin(self.vdf.index)]).unique()
            self._vdf = pd.concat([self.vdf, pd.DataFrame(index=vertex_index)], axis=1).sort_index()

    def remove_vertices(self, vertices=None):
        vertices = list_validator(vertices)
        self._nx_graph.remove_nodes_from(vertices)
        self._vdf = self.vdf.loc[~self._vdf.index.isin(vertices), :]

    @property
    def vertex_attributes(self):
        return self.vdf.columns.tolist()

    def add_vertex_attributes(self, vertex_attributes=None):
        if vertex_attributes is not None:
            if not isinstance(vertex_attributes, dict):
                vertex_attributes = list_validator(vertex_attributes)
                vertex_attributes = {item: None for item in vertex_attributes}
            # vertex_attributes = {key: value for key, value in vertex_attributes.items()
            #                      if key not in self.vdf.columns}
            self._vdf = self.vdf.assign(**vertex_attributes)

    def remove_vertex_attributes(self, vertex_attributes=None):
        vertex_attributes = list_validator(vertex_attributes)
        self._vdf = self._vdf.loc[:, ~self._vdf.columns.isin(vertex_attributes)]

    def add_edges(self, edges=None):
        if edges is not None:
            edges = list_validator(edges)
            if not isinstance(edges[0], (list, tuple)):
                edges = [edges]
            edges = [(edge[0], edge[1]) for edge in edges]
            self.add_vertices(vertices=list(chain.from_iterable(edges)))
            edges = set(edges)
            self._add_modified_edges(edges)
            self._nx_graph.add_edges_from(list(edges))
            self._edges.update(edges)

    @staticmethod
    def _add_modified_edges(edges):
        edges.update({(edge[1], edge[0]) for edge in edges})

    def remove_edges(self, edges=None):
        if edges is not None:
            edges = list_validator(edges)
            if not isinstance(edges[0], (list, tuple)):
                edges = [edges]
            edges = set((edge[0], edge[1]) for edge in edges)
            self._add_modified_edges(edges)
            self._nx_graph.remove_edges_from(list(edges))
            self._edges = self._edges - edges

    @property
    def degree(self):
        return int(np.nan_to_num(np.nanmax([self.edf.sum(axis=1).max(), self.edf.sum(axis=0).max()])))

    def get_node_degrees(self):
        self._vdf['degree'] = self.edf.sum(axis=1)

    @property
    def size(self):
        return len(self.vertices)

    def get_neighbours(self):
        self._vdf['neighbours'] = self.edf.apply(lambda row: row.index[row == 1].tolist(), axis=1)

    def get_second_neighbours(self):
        self.get_neighbours()

        self._vdf['second_neighbours'] = self.vdf['neighbours']\
            .apply(lambda neighbours: set(chain.from_iterable(
                [self.vdf.loc[vertex, 'neighbours'] for vertex in neighbours])))
        self._vdf['second_neighbours'] = self._vdf['second_neighbours'].reset_index().apply(
            lambda row: list(row['second_neighbours'] - {row[vertex_index_name]}), axis=1)

    def get_second_degree(self):
        self.get_second_neighbours()
        self._vdf['second_degree'] = self.vdf['second_neighbours'].apply(lambda neighbours: len(neighbours))

    def draw(self, pos=None, with_labels=None, colour_values=None):
        if pos is True:
            pos = hierarchy_pos(self.nx_graph, 0)
        if with_labels is True:
            labels = self.vdf['label'].todict()
        else:
            labels = None
        nx.draw(self.nx_graph, pos=pos, with_labels=True, labels=labels, node_color=colour_values)

    def from_nx_graph(self, nx_graph):
        self.add_vertices(vertices=list(nx_graph.nodes))
        self.add_edges(edges=list(nx_graph.edges))

    def __len__(self):
        return self.size

    def __str__(self):
        return 'Graph - Vertices: ' + str(self.vertices)

    def __repr__(self):
        return self.__str__()


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