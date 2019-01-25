import pytest
import pandas as pd
from collections import Counter

from graph import Graph


def compare_lists_unordered(list_1, list_2):
    return Counter(list_1) == Counter(list_2)


def test_Graph_init():
    g = Graph()
    assert isinstance(g.vdf, pd.DataFrame) and isinstance(g.edf, pd.DataFrame)


def test_add_vertices():
    g = Graph()
    g.add_vertices()
    assert g.vertices == []
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame())
    g.add_vertices([0])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0]))
    assert g.vertices == [0]
    g.add_vertices(1)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0, 1]))
    assert g.vertices == [0, 1]
    g.add_vertices([3, 2])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0, 1, 2, 3]))
    assert g.vertices == [0, 1, 2, 3]
    g.add_vertices(3)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0, 1, 2, 3]))
    assert g.vertices == [0, 1, 2, 3]

    g = Graph(vertices=[0, 1, 2])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0, 1, 2]))
    assert g.vertices == [0, 1, 2]

    g = Graph()
    g.add_vertices(n_vertices=3)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0, 1, 2]))
    assert g.vertices == [0, 1, 2]

    g = Graph(vertices=[0])
    g.add_vertices(n_vertices=3)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=[0, 1, 2, 3]))
    assert g.vertices == [0, 1, 2, 3]


def test_add_vertex_attributes():
    g = Graph(vertices=[0, 1])
    g.add_vertex_attributes()
    assert g.vertex_attributes == []
    g.add_vertex_attributes(vertex_attributes='name')
    assert g.vertex_attributes == ['name']
    g.add_vertex_attributes(vertex_attributes='level')
    assert compare_lists_unordered(g.vertex_attributes, ['name', 'level'])

    g = Graph(vertices=[0, 1])
    g.add_vertex_attributes(vertex_attributes={'name': 'test', 'level': 0})
    assert compare_lists_unordered(g.vertex_attributes, ['name', 'level'])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame({'name': ['test']*2, 'level': [0]*2}, index=[0, 1]))
    g.add_vertex_attributes(vertex_attributes={'name': 'change_test', 'new_level': 1})
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame({'name': ['change_test']*2, 'level': [0]*2, 'new_level': [1]*2},
                                                      index=[0, 1]))


def test_add_vertices_with_vertex_attributes():
    g = Graph(vertices=[0, 1], vertex_attributes=['name', 'level'])
    g.add_vertices(vertices=[2, 3])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame({'name': [None]*4, 'level': [None]*4}, index=[0, 1, 2, 3]))


def test_update_edf_from_vdf():
    g = Graph()
    g._vdf = pd.DataFrame(index=[0, 1])
    g.update_edf_from_vdf()
    pd.testing.assert_frame_equal(g.edf, pd.DataFrame({0: [0]*2, 1: [0]*2}, index=[0, 1]))

    g._vdf = pd.DataFrame(index=[0, 1, 2, 3])
    g.update_edf_from_vdf()
    pd.testing.assert_frame_equal(g.edf, pd.DataFrame({0: [0]*4, 1: [0]*4, 2: [0]*4, 3: [0]*4}, index=[0, 1, 2, 3]))

    g._edf = pd.DataFrame({0: [0, 1], 1: [1, 0]}, index=[0, 1])
    g.update_edf_from_vdf()
    pd.testing.assert_frame_equal(g.edf, pd.DataFrame({0: [0, 1] + [0]*2, 1: [1, 0] + [0]*2, 2: [0]*4, 3: [0]*4},
                                                      index=[0, 1, 2, 3]))


def test_vdf():

    g = Graph(vertices=[0, 1], vertex_attributes='name')
    g.vdf.loc[0, 'name'] = 'first'
    assert g._vdf.loc[0, 'name'] == 'first' and g.vdf.loc[0, 'name'] == 'first'
