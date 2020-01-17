import pytest
import numpy as np
import motmetrics.lap as lap

SOLVERS = ['scipy', 'munkres', 'ortools', 'lap', 'lapmod', 'lapsolver']
SLOW_SOLVERS = ['scipy', 'munkres']
SPARSE_SOLVERS = ['ortools', 'lapmod']


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_easy(solver):
    """Problem that could be solved by a greedy algorithm."""
    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full(solver):
    """Problem that would be incorrect using a greedy algorithm."""
    costs = np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 6 + 2 + 2.
    expected = np.asfarray([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full_negative(solver):
    costs = -7 + np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 5 + 1 + 1.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_empty(solver):
    costs = np.asfarray([[]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    np.testing.assert_equal(np.size(result), 0)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_infeasible_raises(solver):
    costs = np.asfarray([[np.nan, np.nan, 1], [np.nan, np.nan, 2], [8, 7, 4]])
    with pytest.raises(AssertionError):
        lap.linear_sum_assignment(costs, solver=solver)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_unbalanced_raises(solver):
    costs = np.zeros((3, 4))
    with pytest.raises(AssertionError):
        lap.linear_sum_assignment(costs, solver=solver)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_disallowed(solver):
    costs = np.asfarray([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_non_integer(solver):
    costs = (1. / 9) * np.asfarray([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_attractive_disallowed(solver):
    """Graph contains an attractive edge that cannot be used."""
    costs = np.asfarray([[-10000, -1], [-1, np.nan]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # The optimal solution is (0, 1), (1, 0) for a cost of -2.
    # Ensure that the algorithm does not choose the (0, 0) edge.
    # This would not be a perfect matching.
    expected = np.array([[0, 1], [1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_attractive_broken_ring(solver):
    """Graph contains cheap broken ring and expensive unbroken ring."""
    costs = np.asfarray([[np.nan, 1000, np.nan], [np.nan, 1, 1000], [1000, np.nan, 1]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal solution is (0, 1), (1, 2), (2, 0) with cost 1000 + 1000 + 1000.
    # Solver might choose (0, 0), (1, 1), (2, 2) with cost inf + 1 + 1.
    expected = np.array([[0, 1, 2], [1, 2, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_on_balanced(solver):
    """Use unbalanced solver on balanced problem."""
    costs = np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 6 + 2 + 2.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_wide(solver):
    costs = np.asfarray([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [1, 2]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_tall(solver):
    costs = np.asfarray([[6, 10], [4, 8], [1, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[1, 2], [0, 1]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_disallowed_wide(solver):
    costs = np.asfarray([[np.nan, 11, 8],
                         [8, np.nan, 7]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [2, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_disallowed_tall(solver):
    costs = np.asfarray([[np.nan, 9],
                         [11, np.nan],
                         [8, 7]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 2], [1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_infeasible_raises(solver):
    costs = np.asfarray([[np.nan, np.nan, 1],
                         [np.nan, np.nan, 2],
                         [np.nan, np.nan, 3],
                         [8, 7, 4]])
    with pytest.raises(AssertionError):
        lap.unbalanced_linear_sum_assignment(costs, solver=solver)


@pytest.mark.parametrize('solver', SOLVERS + ['greedy'])
def test_min_weight_negative_easy(solver):
    """Problem that could be solved by a greedy algorithm."""
    costs = -11 + np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_negative(solver):
    """Result should match linear sum assignment."""
    costs = -7 + np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 5 + 1 + 1.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_positive(solver):
    """No edges should be selected."""
    costs = np.asfarray([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[], []])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS + ['greedy'])
def test_min_weight_with_zeros(solver):
    """Zero-cost edges should not be selected."""
    costs = np.asfarray([[0., 0., -1], [0., 0., 0.]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0], [2]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS + ['greedy'])
def test_min_weight_with_negative_zeros(solver):
    """Zero-cost edges should not be selected."""
    costs = np.asfarray([[-0., -0., -1], [-0., -0., -0.]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0], [2]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_wide(solver):
    costs = np.asfarray([[np.nan, -1, -5],
                         [-2, np.nan, -4]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0, 1], [2, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_tall(solver):
    costs = np.asfarray([[np.nan, -2], [-1, np.nan], [-5, -4]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0, 2], [1, 0]])
    np.testing.assert_equal(result, expected)
    np.testing.assert_equal(costs, costs_copy)


def test_change_solver():

    def mysolver(x):
        mysolver.called += 1
        return np.array([]), np.array([])
    mysolver.called = 0

    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])

    with lap.set_default_solver(lap.Solver(mysolver, lap.ASSIGN)):
        rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1

##############
# BENCHMARKS #
##############


@pytest.mark.parametrize('solver', SOLVERS)
def test_benchmark_assign_3x3(benchmark, solver):
    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
    benchmark(lap.linear_sum_assignment, costs, solver=solver)


def random_distance_matrix(rand, size):
    # Take all distance between two sets of 2D points.
    len_x, len_y = size
    x = rand.uniform(0, 1, size=(len_x, 2))
    y = rand.uniform(0, 1, size=(len_y, 2))
    dist = np.linalg.norm(x[:, np.newaxis, :] - y[np.newaxis, :, :], axis=-1)
    return dist


def random_sparse_distance_matrix(rand, size, min_degree):
    # Take all distance between two sets of 2D points.
    dist = random_distance_matrix(rand, size)
    # Find best `min_degree` edges for each vertex.
    # TODO: How to ensure that the assignment problem is feasible?
    num_rows, num_cols = size
    # Index twice to avoid checking whether min_degree > size.
    # Preserve dimensions for broadcasting.
    kth_per_row = np.sort(dist, axis=1)[:, :min_degree][:, -1:]
    kth_per_col = np.sort(dist, axis=0)[:min_degree, :][-1:, :]
    mask = np.logical_or(dist <= kth_per_row, dist <= kth_per_col)
    subset_v = dist[mask]
    subset_i, subset_j = mask.nonzero()
    elems = {(i, j): v for i, j, v in zip(subset_i, subset_j, subset_v)}
    return lap.SparseGraph(size, elems)


def random_sparse_negative_count_matrix(rand, size, sparsity, magnitude=1000):
    costs = -rand.randint(magnitude, size=size)
    mask = (rand.uniform(size=size) < sparsity)
    subset_v = costs[mask]
    subset_i, subset_j = mask.nonzero()
    elems = {(i, j): v for i, j, v in zip(subset_i, subset_j, subset_v)}
    return lap.SparseGraph(size, elems)


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('n', [100])
def test_benchmark_assign_dense_distance_small(benchmark, n, solver):
    rand = np.random.RandomState(0)
    costs = random_distance_matrix(rand, size=(n, n))
    benchmark(lap.linear_sum_assignment, costs, solver=solver)


@pytest.mark.parametrize('solver', set(SOLVERS) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n', [1000])
def test_benchmark_assign_dense_distance_medium(benchmark, n, solver):
    rand = np.random.RandomState(0)
    costs = random_distance_matrix(rand, size=(n, n))
    benchmark(lap.linear_sum_assignment, costs, solver=solver)


@pytest.mark.parametrize('solver', set(SOLVERS) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n,min_degree', [(1000, 20)])
def test_benchmark_assign_sparse_distance_medium(benchmark, n, min_degree, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_distance_matrix(rand, size=(n, n), min_degree=min_degree)
    benchmark(lap.linear_sum_assignment, costs, solver=solver)


@pytest.mark.parametrize('solver', SPARSE_SOLVERS)
@pytest.mark.parametrize('n,min_degree', [(10000, 20)])
def test_benchmark_assign_sparse_distance_large(benchmark, n, min_degree, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_distance_matrix(rand, size=(n, n), min_degree=min_degree)
    benchmark(lap.linear_sum_assignment, costs, solver=solver)


@pytest.mark.parametrize('solver', set(SOLVERS + ['greedy']) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n,sparsity', [(1000, 0.01)])
def test_benchmark_min_weight_sparse_count_medium(benchmark, n, sparsity, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_negative_count_matrix(rand, size=(n, n), sparsity=sparsity)
    benchmark(lap.minimum_weight_matching, costs, solver=solver)


@pytest.mark.parametrize('solver', SPARSE_SOLVERS + ['greedy'])
@pytest.mark.parametrize('n,sparsity', [(10000, 0.001)])
def test_benchmark_min_weight_sparse_count_large(benchmark, n, sparsity, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_negative_count_matrix(rand, size=(n, n), sparsity=sparsity)
    benchmark(lap.minimum_weight_matching, costs, solver=solver)
