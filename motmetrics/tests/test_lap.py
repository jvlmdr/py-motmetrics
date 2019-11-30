import pytest
import itertools
import numpy as np
import motmetrics.lap as lap

SOLVERS = ['ortools', 'scipy', 'munkres', 'lap', 'lapsolver']
SLOW_SOLVERS = ['scipy', 'munkres']
SPARSE_SOLVERS = ['ortools']

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full(solver):
    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_empty(solver):
    costs = np.array([[]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    np.testing.assert_equal(np.size(result), 0)
    np.testing.assert_allclose(costs, costs_copy)

# Note: The munkres solver hangs if the problem is infeasible.
@pytest.mark.parametrize('solver', set(SOLVERS) - set(['munkres']))
def test_assign_infeasible_raises(solver):
    costs = np.array([[np.nan, np.nan, 1], [np.nan, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    with pytest.raises(Exception):
        result = lap.linear_sum_assignment(costs, solver=solver)
        print(result)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_missing_edge(solver):
    costs = np.array([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_non_integer(solver):
    costs = (1. / 9) * np.array([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4.]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_missing_edge_negative(solver):
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
def test_assign_missing_edge_positive(solver):
    """Test that missing edge is not used in assignment."""
    costs = np.array([[np.nan, 1000, np.nan], [np.nan, 1, 1000], [1000, np.nan, 1]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # The optimal solution is (0, 1), (1, 2), (2, 0) with cost 1000 + 1000 + 1000.
    # A bad implementation may choose (0, 0), (1, 1), (2, 2) with cost inf + 1 + 1.
    expected = np.array([[0, 1, 2], [1, 2, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_square(solver):
    costs = np.array([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [1, 2]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_wide(solver):
    costs = np.array([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [1, 2]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_tall(solver):
    costs = np.array([[6, 10], [4, 8], [1, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[1, 2], [0, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_positive(solver):
    costs = np.array([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[], []])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

def test_change_solver():

    def mysolver(x):
        mysolver.called += 1
        return np.array([]), np.array([])
    mysolver.called = 0

    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])

    with lap.set_default_solver(lap.Solver(mysolver, lap.ASSIGN)):
        rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1

######## BENCHMARKS ########

@pytest.mark.parametrize('solver', SOLVERS)
def test_benchmark_assign_3x3(benchmark, solver):
    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4.]])
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

def random_dense(rand, size):
    return rand.uniform(-1, 1, size=size)

@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('n', [100])
def test_benchmark_assign_dense_small(benchmark, n, solver):
    rand = np.random.RandomState(0)
    costs = random_dense(rand, size=(n, n))
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

@pytest.mark.parametrize('solver', set(SOLVERS) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n', [1000])
def test_benchmark_assign_dense_medium(benchmark, n, solver):
    rand = np.random.RandomState(0)
    costs = random_dense(rand, size=(n, n))
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

def random_sparse(rand, size, sparsity):
    # Note: This does not guarantee that the problem will be feasible.
    x = random_dense(rand, size)
    keep = (rand.uniform(size=size) <= sparsity)
    elems = {}
    m, n = size
    for i in range(m):
        for j in range(n):
            if keep[i, j]:
                elems[i, j] = x[i, j]
    return lap.SparseGraph(size, elems)

@pytest.mark.parametrize('represent', ['sparse'])
@pytest.mark.parametrize('solver', set(SOLVERS) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n,sparsity', [(1000, 0.01)])
def test_benchmark_assign_sparse_medium(benchmark, n, sparsity, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse(rand, size=(n, n), sparsity=sparsity)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

@pytest.mark.parametrize('represent', ['sparse', 'dense'])
@pytest.mark.parametrize('solver', SPARSE_SOLVERS)
@pytest.mark.parametrize('n,sparsity', [(10000, 0.001)])
def test_benchmark_assign_sparse_large(benchmark, n, sparsity, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse(rand, size=(n, n), sparsity=sparsity)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

@pytest.mark.parametrize('represent', ['sparse'])
@pytest.mark.parametrize('solver', set(SOLVERS + ['greedy']) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n,sparsity', [(1000, 0.01)])
def test_benchmark_min_weight_sparse_medium(benchmark, n, sparsity, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse(rand, size=(n, n), sparsity=sparsity)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.minimum_weight_matching, costs, solver=solver)

@pytest.mark.parametrize('represent', ['sparse'])
@pytest.mark.parametrize('solver', SPARSE_SOLVERS + ['greedy'])
@pytest.mark.parametrize('n,sparsity', [(10000, 0.001)])
def test_benchmark_min_weight_sparse_large(benchmark, n, sparsity, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse(rand, size=(n, n), sparsity=sparsity)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.minimum_weight_matching, costs, solver=solver)
