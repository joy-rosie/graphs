import pandas as pd
import numpy as np
import networkx as nx


class Graph:

    _vdf = None
    _edf = None

    def initialise(self):
        self._vdf = pd.DataFrame()
        self._edf = pd.DataFrame()

    def __init__(self, vertices=None, vertex_attributes=None):
        self.initialise()
        self.add_vertices(vertices)
        self.add_vertex_attributes(vertex_attributes)

    @property
    def vdf(self):
        return self._vdf

    @vdf.setter
    def vdf(self, value):
        self._vdf = value
        self._vdf = self._vdf.sort_index()
        self.update_edf_from_vdf()

    @property
    def edf(self):
        return self._edf

    @edf.setter
    def edf(self, value):
        self._edf = value.astype('int64')

    @property
    def vertices(self):
        return self.vdf.index.tolist()

    @property
    def edges(self):
        return 'Not implemented'

    def add_vertices(self, vertices=None, n_vertices=None):
        if n_vertices is not None:
            start = int(np.nanmax([self.vdf.index.max(), -1]))
            vertices = list(range(start+1, start + n_vertices + 1))

        if vertices is not None:
            if not isinstance(vertices, list):
                vertices = [vertices]

            new_index = pd.Index(vertices)

            self.vdf = pd.concat([self.vdf, pd.DataFrame(index=new_index[~new_index.isin(self.vdf.index)])],
                                 axis=1)

    @property
    def vertex_attributes(self):
        return self.vdf.columns.tolist()

    def add_vertex_attributes(self, vertex_attributes=None):
        if vertex_attributes is not None:
            if not isinstance(vertex_attributes, dict):
                if not isinstance(vertex_attributes, list):
                    vertex_attributes = [vertex_attributes]
                vertex_attributes = {item: None for item in vertex_attributes}
            # vertex_attributes = {key: value for key, value in vertex_attributes.items()
            #                      if key not in self.vdf.columns}
            self.vdf = self.vdf.assign(**vertex_attributes)

    def update_edf_from_vdf(self):
        new_edf = pd.DataFrame(0, index=self.vdf.index, columns=self.vdf.index)
        new_edf.update(self.edf)
        self.edf = new_edf

    def __str__(self):
        return 'Graph - Vertices: ' + str(self.vertices)

    def __repr__(self):
        return self.__str__()
