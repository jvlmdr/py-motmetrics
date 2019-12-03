import pytest
import itertools
import numpy as np
import motmetrics.lap as lap

SOLVERS = ['ortools', 'scipy', 'munkres', 'lap', 'lapmod', 'lapsolver']
SLOW_SOLVERS = ['scipy', 'munkres']
SPARSE_SOLVERS = ['ortools', 'lapmod']

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_easy(solver):
    """Problem that could be solved by a greedy algorithm."""
    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full(solver):
    """Problem that would be incorrect using a greedy algorithm."""
    costs = np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 6 + 2 + 2.
    expected = np.asfarray([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_full_negative(solver):
    costs = -7 + np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 5 + 1 + 1.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_empty(solver):
    costs = np.asfarray([[]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    np.testing.assert_equal(np.size(result), 0)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_infeasible_raises(solver):
    costs = np.asfarray([[np.nan, np.nan, 1], [np.nan, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    with pytest.raises(Exception):
        result = lap.linear_sum_assignment(costs, solver=solver)
        print(result)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_disallowed(solver):
    costs = np.asfarray([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_non_integer(solver):
    costs = (1. / 9) * np.asfarray([[5, 9, np.nan], [10, np.nan, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1, 2], [0, 2, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

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
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_assign_attractive_broken_ring(solver):
    """Graph contains cheap broken ring and expensive unbroken ring."""
    costs = np.asfarray([[np.nan, 1000, np.nan], [np.nan, 1, 1000], [1000, np.nan, 1]])
    costs_copy = costs.copy()
    result = lap.linear_sum_assignment(costs, solver=solver)

    # The optimal solution is (0, 1), (1, 2), (2, 0) with cost 1000 + 1000 + 1000.
    # A bad implementation may choose (0, 0), (1, 1), (2, 2) with cost inf + 1 + 1.
    expected = np.array([[0, 1, 2], [1, 2, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_on_balanced(solver):
    """Use unbalanced solver on balanced problem."""
    costs = np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 6 + 2 + 2.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_wide(solver):
    costs = np.asfarray([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [1, 2]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_tall(solver):
    costs = np.asfarray([[6, 10], [4, 8], [1, 2]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[1, 2], [0, 1]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_disallowed_wide(solver):
    costs = np.asfarray([[np.nan, 11, 8],
                         [8, np.nan, 7]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 1], [2, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_disallowed_tall(solver):
    costs = np.asfarray([[np.nan, 9],
                         [11, np.nan],
                         [8, 7]])
    costs_copy = costs.copy()
    result = lap.unbalanced_linear_sum_assignment(costs, solver=solver)

    expected = np.array([[0, 2], [1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_unbalanced_infeasible_raises(solver):
    costs = np.asfarray([[np.nan, np.nan, 1],
                         [np.nan, np.nan, 2],
                         [np.nan, np.nan, 3],
                         [8, 7, 4]])
    costs_copy = costs.copy()
    with pytest.raises(Exception):
        result = lap.unbalanced(costs, solver=solver)
        print(result)

@pytest.mark.parametrize('solver', SOLVERS + ['greedy'])
def test_min_weight_negative_easy(solver):
    """Problem that could be solved by a greedy algorithm."""
    costs = -11 + np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_negative(solver):
    """Result should match linear sum assignment."""
    costs = -7 + np.asfarray([[5, 5, 6], [1, 2, 5], [2, 4, 5]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    # Optimal matching is (0, 2), (1, 1), (2, 0) for 5 + 1 + 1.
    expected = np.array([[0, 1, 2], [2, 1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_positive(solver):
    """No edges should be selected."""
    costs = np.asfarray([[6, 4, 1], [10, 8, 2]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[], []])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS + ['greedy'])
def test_min_weight_with_zeros(solver):
    """Zero-cost edges should not be selected."""
    costs = np.asfarray([[0., 0., -1], [0., 0., 0.]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0], [2]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS + ['greedy'])
def test_min_weight_with_negative_zeros(solver):
    """Zero-cost edges should not be selected."""
    costs = np.asfarray([[-0., -0., -1], [-0., -0., -0.]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0], [2]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_wide(solver):
    costs = np.asfarray([[np.nan, -1, -5],
                         [-2, np.nan, -4]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0, 1], [2, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

@pytest.mark.parametrize('solver', SOLVERS)
def test_min_weight_tall(solver):
    costs = np.asfarray([[np.nan, -2], [-1, np.nan], [-5, -4]])
    costs_copy = costs.copy()
    result = lap.minimum_weight_matching(costs, solver=solver)

    expected = np.array([[0, 2], [1, 0]])
    np.testing.assert_allclose(result, expected)
    np.testing.assert_allclose(costs, costs_copy)

def test_change_solver():

    def mysolver(x):
        mysolver.called += 1
        return np.array([]), np.array([])
    mysolver.called = 0

    costs = np.array([[6, 9, 1], [10, 3, 2], [8, 7, 4]])

    with lap.set_default_solver(lap.Solver(mysolver, lap.ASSIGN)):
        rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1
    rids, cids = lap.linear_sum_assignment(costs)
    assert mysolver.called == 1

######## BENCHMARKS ########

@pytest.mark.parametrize('solver', SOLVERS)
def test_benchmark_assign_3x3(benchmark, solver):
    costs = np.asfarray([[6, 9, 1], [10, 3, 2], [8, 7, 4]])
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

def random_sparse_min_degree(rand, size, min_degree):
    """Generates a graph with degree in [min_degree, 2 * min_degree]."""
    x = random_dense(rand, size)
    m, n = size
    elems = {}
    for i in range(m):
        subset = rand.choice(n, size=min_degree, replace=False)
        for j in subset:
            elems[i, j] = x[i, j]
    for j in range(n):
        subset = rand.choice(m, size=min_degree, replace=False)
        for i in subset:
            elems[i, j] = x[i, j]
    return lap.SparseGraph(size, elems)

def random_sparse(rand, size, sparsity):
    """Does not guarantee that the graph will be connected."""
    x = random_dense(rand, size)
    keep = (rand.uniform(size=size) <= sparsity)
    elems = {}
    m, n = size
    for i in range(m):
        for j in range(n):
            if keep[i, j]:
                elems[i, j] = x[i, j]
    return lap.SparseGraph(size, elems)

def choose_sparsity(n, prob_connected):
    """Chooses sparsity for [n, n] matrix which satisfies p(connected)."""
    # Let s be a sparsity factor.
    # p(empty row) = (1 - s) ** n
    # q = p(no empty rows) = (1 - (1 - s) ** n) ** n
    # 1 - q ** (1/n) = (1 - s) ** n
    # 1 - [1 - q ** (1/n)] ** (1/n) = s
    return 1 - (1 - prob_connected ** (1 / n)) ** (1 / n)

@pytest.mark.parametrize('represent', ['sparse'])
@pytest.mark.parametrize('solver', set(SOLVERS) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n,min_degree', [(1000, 10)])
def test_benchmark_assign_sparse_medium(benchmark, n, min_degree, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_min_degree(rand, size=(n, n), min_degree=min_degree)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

@pytest.mark.parametrize('represent', ['sparse', 'dense'])
@pytest.mark.parametrize('solver', SPARSE_SOLVERS)
@pytest.mark.parametrize('n,min_degree', [(10000, 10)])
def test_benchmark_assign_sparse_large(benchmark, n, min_degree, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_min_degree(rand, size=(n, n), min_degree=min_degree)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.linear_sum_assignment, costs, solver=solver)

@pytest.mark.parametrize('represent', ['sparse'])
@pytest.mark.parametrize('solver', set(SOLVERS + ['greedy']) - set(SLOW_SOLVERS))
@pytest.mark.parametrize('n,min_degree', [(1000, 10)])
def test_benchmark_min_weight_sparse_medium(benchmark, n, min_degree, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_min_degree(rand, size=(n, n), min_degree=min_degree)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.minimum_weight_matching, costs, solver=solver)

@pytest.mark.parametrize('represent', ['sparse'])
@pytest.mark.parametrize('solver', SPARSE_SOLVERS + ['greedy'])
@pytest.mark.parametrize('n,min_degree', [(10000, 10)])
def test_benchmark_min_weight_sparse_large(benchmark, n, min_degree, represent, solver):
    rand = np.random.RandomState(0)
    costs = random_sparse_min_degree(rand, size=(n, n), min_degree=min_degree)
    if represent == 'dense':
        costs = lap.sparse2dense(costs)
    benchmark(lap.minimum_weight_matching, costs, solver=solver)
