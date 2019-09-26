import numpy as np

from flatland.core.grid.grid_utils import Vec2dOperations as Vec2d
from flatland.core.grid.rail_env_grid import RailEnvTransitions
from flatland.core.transition_map import GridTransitionMap
from flatland.envs.grid4_generators_utils import connect_rail, connect_nodes, connect_from_nodes, connect_to_nodes


def test_build_railway_infrastructure():
    rail_trans = RailEnvTransitions()
    grid_map = GridTransitionMap(width=20, height=20, transitions=rail_trans)
    grid_map.grid.fill(0)
    np.random.seed(0)

    start_point = (2, 2)
    end_point = (8, 8)
    connection_001 = connect_rail(rail_trans, grid_map, start_point, end_point, Vec2d.get_manhattan_distance)
    connection_001_expected = [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6), (2, 7), (2, 8), (3, 8), (4, 8), (5, 8), (6, 8),
                               (7, 8), (8, 8)]

    start_point = (1, 3)
    end_point = (1, 7)
    connection_002 = connect_nodes(rail_trans, grid_map, start_point, end_point, Vec2d.get_manhattan_distance)
    connection_002_expected = [(1, 3), (1, 4), (1, 5), (1, 6), (1, 7)]

    start_point = (6, 2)
    end_point = (6, 5)
    connection_003 = connect_from_nodes(rail_trans, grid_map, start_point, end_point, Vec2d.get_manhattan_distance)
    connection_003_expected = [(6, 2), (6, 3), (6, 4), (6, 5)]

    start_point = (7, 5)
    end_point = (8, 9)
    connection_004 = connect_to_nodes(rail_trans, grid_map, start_point, end_point, Vec2d.get_manhattan_distance)
    connection_004_expected = [(7, 5), (7, 6), (7, 7), (7, 8), (7, 9), (8, 9)]

    assert connection_001 == connection_001_expected, \
        "actual={}, expected={}".format(connection_001, connection_001_expected)
    assert connection_002 == connection_002_expected, \
        "actual={}, expected={}".format(connection_002, connection_002_expected)
    assert connection_003 == connection_003_expected, \
        "actual={}, expected={}".format(connection_003, connection_003_expected)
    assert connection_004 == connection_004_expected, \
        "actual={}, expected={}".format(connection_004, connection_004_expected)

    grid_map_grid_expected = [
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1025, 1025, 1025, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 4, 1025, 1025, 1025, 1025, 1025, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1025, 1025, 256, 0, 0, 32800, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 4, 1025, 1025, 33825, 4608, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 128, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ]

    assert np.all(grid_map.grid == grid_map_grid_expected), \
        "actual={}, expected={}".format(grid_map.grid, grid_map_grid_expected)