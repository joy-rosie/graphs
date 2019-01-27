import pytest
import pandas as pd
import networkx as nx
from collections import Counter
import random

from graph import Graph
from graph import make_vertex_index


def compare_lists_unordered(list_1, list_2):
    return Counter(list_1) == Counter(list_2)


def test_Graph_init():
    g = Graph()
    assert isinstance(g.vdf, pd.DataFrame) and isinstance(g.edf, pd.DataFrame)


def test_add_vertices():
    g = Graph()
    g.add_vertices()
    assert g.vertices == []
    assert compare_lists_unordered(list(g.nx_graph.nodes), [])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index()))
    g.add_vertices([0])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0])))
    assert g.vertices == [0]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0])
    g.add_vertices(1)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0, 1])))
    assert g.vertices == [0, 1]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0, 1])
    g.add_vertices([3, 2])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0, 1, 2, 3])))
    assert g.vertices == [0, 1, 2, 3]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0, 1, 2, 3])
    g.add_vertices(3)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0, 1, 2, 3])))
    assert g.vertices == [0, 1, 2, 3]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0, 1, 2, 3])

    g = Graph(vertices=[0, 1, 2])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0, 1, 2])))
    assert g.vertices == [0, 1, 2]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0, 1, 2])

    g = Graph()
    g.add_vertices(n_vertices=3)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0, 1, 2])))
    assert g.vertices == [0, 1, 2]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0, 1, 2])

    g = Graph(vertices=[0])
    g.add_vertices(n_vertices=3)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([0, 1, 2, 3])))
    assert g.vertices == [0, 1, 2, 3]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [0, 1, 2, 3])


def test_remove_vertices():
    g = Graph()
    g.remove_vertices()
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index()))
    assert g.vertices == []
    assert compare_lists_unordered(list(g.nx_graph.nodes), [])

    g.remove_vertices(vertices=[0, 1])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index()))
    assert g.vertices == []
    assert compare_lists_unordered(list(g.nx_graph.nodes), [])

    g = Graph(vertices=[0, 1, 2])
    g.remove_vertices(vertices=[0, 1])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index([2])))
    assert g.vertices == [2]
    assert compare_lists_unordered(list(g.nx_graph.nodes), [2])

    g.remove_vertices(vertices=2)
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index()))
    assert g.vertices == []
    assert compare_lists_unordered(list(g.nx_graph.nodes), [])

    g = Graph(n_vertices=10)
    g.remove_vertices(vertices=list(range(10)))
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index()))
    assert g.vertices == []
    assert compare_lists_unordered(list(g.nx_graph.nodes), [])

    g = Graph(n_vertices=10)
    g.remove_vertices(vertices=list(range(100)))
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame(index=make_vertex_index()))
    assert g.vertices == []
    assert compare_lists_unordered(list(g.nx_graph.nodes), [])


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
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame({'name': ['test']*2, 'level': [0]*2},
                                                      index=make_vertex_index([0, 1])))
    g.add_vertex_attributes(vertex_attributes={'name': 'change_test', 'new_level': 1})
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame({'name': ['change_test']*2, 'level': [0]*2, 'new_level': [1]*2},
                                                      index=make_vertex_index([0, 1])))


def test_add_vertices_with_vertex_attributes():
    g = Graph(vertices=[0, 1], vertex_attributes=['name', 'level'])
    g.add_vertices(vertices=[2, 3])
    pd.testing.assert_frame_equal(g.vdf, pd.DataFrame({'name': [None]*4, 'level': [None]*4},
                                                      index=make_vertex_index([0, 1, 2, 3])))


def test_add_edges():

    g = Graph(vertices=0)

    g.add_edges(edges=[0, 0])
    pd.testing.assert_frame_equal(g.edf, pd.DataFrame(1, index=[0], columns=[0]))
    assert compare_lists_unordered(list(g.nx_graph.edges), [(0, 0)])

    g.add_edges(edges=(1, 0))
    pd.testing.assert_frame_equal(g.edf, pd.DataFrame({0: [1, 1], 1: [1, 0]}, index=[0, 1], columns=[0, 1]))
    assert g.vertices == [0, 1]
    assert compare_lists_unordered(list(g.nx_graph.edges), [(0, 0), (0, 1)])

    g.add_edges(edges=[(0, 1), [1, 1]])
    pd.testing.assert_frame_equal(g.edf, pd.DataFrame({0: [1, 1], 1: [1, 1]}, index=[0, 1], columns=[0, 1]))
    assert compare_lists_unordered(list(g.nx_graph.edges), [(0, 0), (0, 1), (1, 1)])


def test_remove_edges():

    g = Graph(edges=[[0, 0]])
    g.remove_edges(edges=[[0, 0]])
    assert g.edges == set() and g.vertices == [0]
    assert compare_lists_unordered(list(g.nx_graph.edges), [])

    g = Graph(edges=[[1, 0]])
    g.remove_edges(edges=[[0, 1]])
    assert g.edges == set() and g.vertices == [0, 1]
    assert compare_lists_unordered(list(g.nx_graph.edges), [])

    g = Graph(edges=[[0, 0], [0, 1]])
    g.remove_edges(edges=[[1, 0]])
    assert g.edges == {(0, 0)} and g.vertices == [0, 1]
    assert compare_lists_unordered(list(g.nx_graph.edges), [(0, 0)])

    g = Graph(edges=[[0, 0], [0, 1], [1, 1]])
    g.remove_edges(edges=[(0, 0), [1, 0]])
    assert g.edges == {(1, 1)} and g.vertices == [0, 1]
    assert compare_lists_unordered(list(g.nx_graph.edges), [(1, 1)])


def test_size_len():
    g = Graph()
    assert g.size == 0 and len(g) == 0

    g = Graph(n_vertices=4)
    assert g.size == 4 and len(g) == 4

    g.remove_vertices(vertices=[2, 3])
    assert g.size == 2 and len(g) == 2

    g.add_edges(edges=[0, 2])
    assert g.size == 3 and len(g) == 3

    g.remove_vertices(vertices=[0, 1, 2])
    assert g.size == 0 and len(g) == 0


def test_degree():
    g = Graph()
    assert g.degree == 0

    g = Graph(edges=[0, 0])
    assert g.degree == 1

    g.add_edges(edges=[0, 1])
    assert g.degree == 2

    g.remove_edges(edges=[0, 1])
    assert g.degree == 1

    g.remove_edges(edges=[0, 0])
    assert g.degree == 0


def test_get_node_degrees():
    g = Graph(vertices=[0, 1, 2, 3, 4], edges=[(0, 1), (1, 2)])
    g.get_node_degrees()
    expected_df = pd.DataFrame({'degree': [1, 2, 1, 0, 0]}, index=make_vertex_index(list(range(5))))
    pd.testing.assert_series_equal(g.vdf['degree'], expected_df['degree'], check_dtype=False)


def test_get_neighbours():
    g = Graph(vertices=[0, 1, 2, 3, 4], edges=[(0, 1), (1, 2)])
    g.get_neighbours()
    expected_df = pd.DataFrame({'neighbours': [[1], [0, 2], [1], [], []]}, index=make_vertex_index(list(range(5))))
    pd.testing.assert_series_equal(g.vdf['neighbours'], expected_df['neighbours'])


def test_get_second_neighbours():
    g = Graph(vertices=[0, 1, 2, 3, 4], edges=[(0, 1), (1, 2)])
    g.get_second_neighbours()
    expected_df = pd.DataFrame({'second_neighbours': [[2], [], [0], [], []]}, index=make_vertex_index(list(range(5))))
    pd.testing.assert_series_equal(g.vdf['second_neighbours'], expected_df['second_neighbours'])


def test_get_second_degree():
    g = Graph(vertices=[0, 1, 2, 3, 4], edges=[(0, 1), (1, 2)])
    g.get_second_degree()
    expected_df = pd.DataFrame({'second_degree': [1, 0, 1, 0, 0]}, index=make_vertex_index(list(range(5))))
    pd.testing.assert_series_equal(g.vdf['second_degree'], expected_df['second_degree'])


def test_draw():
    g = Graph(vertices=[0, 1, 2, 3, 4], edges=[(0, 1), (1, 2)])
    g.draw()


def test_from_nx_graph():
    for _ in range(10):
        nx_graph = nx.random_tree(10)
        g = Graph()
        g.from_nx_graph(nx_graph=nx_graph)
        assert compare_lists_unordered(g.vertices, list(nx_graph.nodes))
        assert compare_lists_unordered(g.edges, list(set(nx_graph.edges).union({(edge[1], edge[0]) for edge in nx_graph.edges})))


def test_add_level():
    g = Graph(n_vertices=1, vertex_attributes='level')
    g._vdf.loc[0, 'level'] = 1
    random.seed(0)
    g.add_random_level(degree=3)
    assert g.vertices == [0, 1, 2]
    assert compare_lists_unordered(g.edges, [(0, 1), (1, 0), (0, 2), (2, 0)])

    g.add_random_level(degree=0)
    assert g.vertices == [0, 1, 2]
    assert compare_lists_unordered(g.edges, [(0, 1), (1, 0), (0, 2), (2, 0)])

    g.add_random_level(degree=1)
    assert g.vertices == [0, 1, 2]
    assert compare_lists_unordered(g.edges, [(0, 1), (1, 0), (0, 2), (2, 0)])

    g.add_random_level(degree=2)
    assert g.vertices == [0, 1, 2, 3, 4]
    assert compare_lists_unordered(g.edges, [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1), (2, 4), (4, 2)])


def test_make_random_tree():
    g = Graph()
    random.seed(0)
    g.make_random_tree(n_levels=1, degree=3)
    assert g.vertices == [0, 1, 2, 3]
    assert compare_lists_unordered(g.edges, [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0)])

    g = Graph()
    g.make_random_tree(n_levels=2, degree=3)
    assert g.vertices == [0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert compare_lists_unordered(g.edges, [(0, 1), (1, 0), (0, 2), (2, 0), (0, 3), (3, 0), (1, 4), (4, 1), (1, 5),
                                             (5, 1), (2, 6), (6, 2), (2, 7), (7, 2), (3, 8), (8, 3)])
