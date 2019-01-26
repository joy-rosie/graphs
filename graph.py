import pandas as pd
import numpy as np
import networkx as nx
from itertools import chain


def list_validator(input_data):
    output = input_data
    if not isinstance(output, list):
        output = [output]
    return output


def make_vertex_index(input_list=None):
    if input_list is None:
        input_list = list()
    return pd.Index(input_list, name='vertex_index', dtype='int64')


class Graph:

    _vdf = None
    _edges = None
    _defaults_for_attributes = None

    def initialise(self):
        self._vdf = pd.DataFrame(index=make_vertex_index())
        self._edges = set()
        self._defaults_for_attributes = dict()

    def __init__(self, vertices=None, n_vertices=None, vertex_attributes=None, edges=None):
        self.initialise()
        self.add_vertices(vertices=vertices, n_vertices=n_vertices)
        self.add_vertex_attributes(vertex_attributes=vertex_attributes)
        self.add_edges(edges=edges)

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

            new_index = pd.Index(vertices)
            vertex_index = make_vertex_index(new_index[~new_index.isin(self.vdf.index)]).unique()
            self._vdf = pd.concat([self.vdf, pd.DataFrame(index=vertex_index)], axis=1).sort_index()

    def remove_vertices(self, vertices=None):
        vertices = list_validator(vertices)
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
            self._edges.update(edges)

    @staticmethod
    def _add_modified_edges(edges):
        edges .update({(edge[1], edge[0]) for edge in edges})

    def remove_edges(self, edges=None):
        if edges is not None:
            edges = list_validator(edges)
            if not isinstance(edges[0], (list, tuple)):
                edges = [edges]
            edges = set((edge[0], edge[1]) for edge in edges)
            self._add_modified_edges(edges)
            self._edges = self._edges - edges

    @property
    def degree(self):
        return int(np.nan_to_num(np.nanmax([self.edf.sum(axis=1).max(), self.edf.sum(axis=0).max()])))

    @property
    def size(self):
        return len(self.vertices)

    def __len__(self):
        return self.size

    def __str__(self):
        return 'Graph - Vertices: ' + str(self.vertices)

    def __repr__(self):
        return self.__str__()
