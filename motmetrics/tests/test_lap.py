import pytest
import numpy as np
import motmetrics.lap as lap

# Note: lapsolver solves MWM (Minimum-Weight Matching) not assignment.
SOLVERS = ['ortools', 'scipy', 'munkres', 'lap']  # 'lapsolver'

@pytest.mark.parametrize('solver', SOLVERS)
def test_lap_full(solver):
    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_lap_missing_edge(solver):
    costs = np.array([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_lap_non_integer(solver):
    costs = (1. / 9) * np.array([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_lap_missing_edge_negative(solver):
    """Test that missing edge is not used in assignment."""
    costs = np.array([[-10000, -1], [-1, np.nan]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # The optimal solution is (0, 1), (1, 0) for a cost of -2.
    # Ensure that the algorithm does not choose the (0, 0) edge.
    # This would not be a perfect matching.
    expected = np.array([[0, 1], [1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_lap_missing_edge_positive(solver):
    """Test that missing edge is not used in assignment."""
    costs = np.array([[np.nan, 1000, np.nan], [np.nan, 1, 1000], [1000, np.nan, 1]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # The optimal solution is (0, 1), (1, 2), (2, 0) with cost 1000 + 1000 + 1000.
    # A bad implementation may choose (0, 0), (1, 1), (2, 2) with cost inf + 1 + 1.
    expected = np.array([[0, 1, 2], [1, 2, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

def test_change_solver():

    def mysolver(x):
        mysolver.called += 1
        return np.array([]), np.array([])
    mysolver.called = 0

    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])

    with lap.set_default_solver(mysolver):
        rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1

