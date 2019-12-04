"""Algorithms for solving assignment problems.

The package supports three problems:
    minimum_weight_matching (MIN_WEIGHT)
    unbalanced_linear_sum_assignment (UNBAL)
    linear_sum_assignment (ASSIGN)

MIN_WEIGHT: Arbitrary matching between sets of different sizes.
UNBAL: One-sided perfect matching between two sets of different sizes.
ASSIGN: Perfect matching between two sets of equal sizes.

UNBAL is a superset of ASSIGN.
Most packages solve ASSIGN, but some support UNBAL.

MIN_WEIGHT can be transformed into an UNBAL problem.
UNBAL can be transformed into an ASSIGN problem.
Therefore MIN_WEIGHT <= UNBAL <= ASSIGN in terms of hardness.
The number of nodes and edges is linear (with respect to
the original number of nodes and edges) for both transformations.

MIN_WEIGHT solvers can only be used to solve MIN_WEIGHT problems.
UNBAL and ASSIGN solvers can be used to solve any of the three problems.

If the graph is complete and contains only negative- and zero-weight edges,
then MIN_WEIGHT is equivalent to UNBAL.
However, the zero-weight edges can be removed in the MIN_WEIGHT problem.
When the graph contains several negative edges and many zero edges,
this can make sparse MIN_WEIGHT much faster than dense UNBAL.

The costs matrix can always be specified using np.ndarray or lap.SparseGraph.
"""

import numpy as np
from contextlib import contextmanager
import six
import warnings

ASSIGN = 'assign'
UNBAL = 'unbal'
MIN_WEIGHT = 'min_weight'

class Solver(object):
    """Adds metadata to function for solving assigment problems."""

    def __init__(self, fn, problem, module=None):
        """Creates a solver object.

        Args:
            fn: Function that maps costs (np.ndarray or lap.SparseGraph)
                to matches describes as tuple (rids, cids).
            problem: {ASSIGN, UNBAL, MIN_WEIGHT}
            module: If this module can be imported, the solver is available
                (string or None).
        """
        self.fn = fn
        self.problem = problem
        self.module = module

    def __call__(self, costs, **kwargs):
        return self.fn(costs, **kwargs)

def _assert_problem_is_valid(problem):
    if problem not in [ASSIGN, UNBAL, MIN_WEIGHT]:
        raise AssertionError('unknown problem type', problem)

def _cost_is_edge(cost):
    """Does this cost represent an edge?"""
    # TODO: Are -inf values supported or not?
    return ~(np.isnan(cost) | np.equal(cost, np.inf))

def minimum_weight_matching(costs, solver=None):
    """Solves the minimum-weight matching (MIN_WEIGHT) problem.

    Finds the matching between two sets with the minimum cost. Unlike the problem
    of linear sum assignment (ASSIGN), some vertices may remain unmatched.
    See: https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf (sec 1.6)

    Args:
        costs: Either np.ndarray or lap.SparseGraph.

    Let r be min(costs.shape), n = max(costs.shape) and m be the number of edges.
    It is converted into UNBAL with size:
        r' = r
        n' = n + r
        m' = m + r
    This may subsequently be converted into ASSIGN with size
        r'' = n'' = n + 2 r
        m'' = 2 m + n + 3 r
    These sizes are linear in the original dimensions.
    """
    solver = _get_solver(solver)
    _assert_problem_is_valid(solver.problem)
    if solver.problem == MIN_WEIGHT:
        # Use the solver directly if it is a MIN_WEIGHT solver.
        rids, cids = solver(costs)
    else:
        # Convert the problem to UNBAL.
        rids, cids = _solve_min_weight_as_unbal(costs, solver=solver)

    matches = list(zip(rids, cids))
    # Exclude edges with zero cost.
    matches = [(i, j) for i, j in matches if costs[i, j] != 0]
    # Sort indices before returning.
    matches = sorted(matches)
    rids, cids = zip(*matches) if matches else ([], [])
    return rids, cids

def _solve_min_weight_as_unbal(costs, solver=None):
    """Converts MIN_WEIGHT into UNBAL and solves it.

    Let r be the size of the smaller set X and n be that of the larger set Y.
    For each node in X, add a zero-weight edge to a new node in set Y.
    For each node in X, we can choose this edge or an existing edge.
    The problem will be unbalanced since we add nodes to the larger set.

    Note: Result is not sorted.
    """
    # Ensure that the first set is the small one.
    use_transpose = (costs.shape[0] > costs.shape[1])
    if use_transpose:
        costs = costs.transpose()
    len_x, len_y = costs.shape

    lsa_shape = (len_x, len_y + len_x)
    if isinstance(costs, SparseGraph):
        # Copy all edges.
        elems = dict(costs.elems)
        # Add new edges with zero weight.
        for i in range(len_x):
            elems[i, len_y + i] = 0
        lsa_costs = SparseGraph(lsa_shape, elems)
    elif isinstance(costs, np.ndarray):
        # Same operation for dense ararys.
        lsa_costs = np.full(lsa_shape, np.nan)
        lsa_costs[:, :len_y] = costs
        np.fill_diagonal(lsa_costs[:, len_y:], 0)
    else:
        raise ValueError('unknown matrix type', type(costs))

    lsa_rids, lsa_cids = unbalanced_linear_sum_assignment(
            lsa_costs, solver=solver)
    # Select subset of matches (i, j) where j is within Y.
    matches = [(i, j) for i, j in zip(lsa_rids, lsa_cids) if j < len_y]

    rids, cids = zip(*matches) if matches else ([], [])
    if use_transpose:
        rids, cids = cids, rids
    return rids, cids

def unbalanced_linear_sum_assignment(costs, solver=None):
    """Solves linear sum assignment where the two sets have different sizes.

    The problem must be "one-sided perfect". That is, there must exist a matching
    (set of edges) where every node in the smaller set has a match (edge).

    Converts to a balanced problem using the doubling reduction.
    See: https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf (sec 1.3)

    Let r be min(costs.shape), n = max(costs.shape) and m be the number of edges.
    If the solver is not an UNBAL solver, then the problem is converted into ASSIGN with
        r' = n' = n + r
        m' = 2 m + n
    The problem size remains linear in the input size.

    Args:
        costs: Either np.ndarray or lap.SparseGraph.
    """
    solver = _get_solver(solver)
    _assert_problem_is_valid(solver.problem)
    if solver.problem == UNBAL:
        # Use the solver directly if it is an UNBAL solver.
        rids, cids = solver(costs)
    else:
        # Otherwise convert to ASSIGN.
        rids, cids = _solve_unbal_as_assign(costs, solver)

    matches = list(zip(rids, cids))
    # Sort indices before returning.
    matches = sorted(matches)
    rids, cids = zip(*matches) if matches else ([], [])
    return rids, cids

def _solve_unbal_as_assign(costs, solver):
    """Converts UNBAL into ASSIGN and solves it.

    Construct a new problem with two sets V and V' of size (n + r).
    Partition V into X = [0, len_x) and Y = len_x + [0, len_y).
    Partition V' into Y' = [0, len_y) and X' = len_y + [0, len_x).
    See: https://www.hpl.hp.com/techreports/2012/HPL-2012-40R1.pdf (sec 1.3)

    Note: Result is not sorted.
    """
    # Ensure that the first set is the small one.
    use_transpose = (costs.shape[0] > costs.shape[1])
    if use_transpose:
        costs = costs.transpose()
    len_x, len_y = costs.shape

    # TODO: Different method for dense matrices?
    # When using a dense method to solve a problem of size [r, n], is it better
    # to (a) pad to [n, n] with zeros or (b) construct [n + r, n + r] problem.
    # Dense methods may still have complexity depending on number of edges?

    bal_shape = (len_x + len_y, len_y + len_x)
    if isinstance(costs, SparseGraph):
        # Start with original set of edges (X, i), (Y', j).
        elems = dict(costs.elems)
        # Add duplicate edges (Y, j), (X', i)
        for (i, j), cost in costs.elems.items():
            elems[len_x + j, len_y + i] = cost
        # Add zero-cost edges for large-to-large connections (Y, j), (Y', j).
        for j in range(len_y):
            elems[len_x + j, j] = 0
        bal_costs = SparseGraph(bal_shape, elems)
    elif isinstance(costs, np.ndarray):
        bal_costs = np.full(bal_shape, np.nan)
        bal_costs[:len_x, :len_y] = costs
        bal_costs[len_x:, len_y:] = costs.transpose()
        np.fill_diagonal(bal_costs[len_x:, :len_y], 0)
    else:
        raise ValueError('unknown matrix type', type(costs))

    bal_rids, bal_cids = linear_sum_assignment(bal_costs, solver=solver)
    # Take subset of edges in X and Y'.
    matches = [
            (i, j) for i, j in zip(bal_rids, bal_cids) if i < len_x and j < len_y
    ]

    rids, cids = zip(*matches) if matches else ([], [])
    if use_transpose:
        rids, cids = cids, rids
    return rids, cids

def linear_sum_assignment(costs, solver=None):
    """Solve a linear sum assignment problem (LSA).

    For large datasets solving the minimum cost assignment becomes the dominant runtime part. 
    We therefore support various solvers out of the box (currently lapsolver, scipy, ortools, munkres)
    
    Params
    ------
    costs : np.array
        numpy matrix containing costs. Use NaN/Inf values for unassignable
        row/column pairs.
    
    Kwargs
    ------
    solver : callable or str, optional
        When str: name of solver to use.
        When callable: function to invoke
        When None: uses first available solver
    """
    if not all(costs.shape):
        return [], []
    if costs.shape[0] != costs.shape[1]:
        raise AssertionError('problem is not balanced', costs.shape)

    solver = _get_solver(solver)
    _assert_problem_is_valid(solver.problem)
    if solver.problem not in [UNBAL, ASSIGN]:
        raise AssertionError('solver problem is not in {UNBAL, ASSIGN}')

    rids, cids = solver(costs)
    return rids, cids

def add_expensive_edges(costs):
    """Replaces non-edge costs (nan, inf) with large number.

    If the optimal solution includes one of these edges,
    then the original problem was infeasible.

    Args:
        costs: np.ndarray
    """
    # The graph is probably already dense if we are doing this.
    assert isinstance(costs, np.ndarray)
    # The linear_sum_assignment function in scipy does not support missing edges.
    # Replace nan with a large constant that ensures it is not chosen.
    # If it is chosen, that means the problem was infeasible.
    valid = _cost_is_edge(costs)
    if valid.all():
        return costs.copy()
    if not valid.any():
        return np.zeros_like(costs)
    r = min(costs.shape)
    # Assume all edges costs are within [-c, c], c >= 0.
    # The cost of an invalid edge must be such that...
    # choosing this edge once and the best-possible edge (r - 1) times
    # is worse than choosing the worst-possible edge r times.
    # l + (r - 1) (-c) > r c
    # l > r c + (r - 1) c
    # l > (2 r - 1) c
    # Choose l = 2 r c + 1 > (2 r - 1) c.
    c = np.abs(costs[valid]).max() + 1  # Doesn't hurt to add 1 here.
    large_constant = 2 * r * c + 1
    return np.where(valid, costs, large_constant)

def lsa_solve_scipy(costs):
    """Solves the LSA problem using the scipy library."""
    from scipy.optimize import linear_sum_assignment as scipy_solve

    costs = _as_dense(costs)
    finite_costs = add_expensive_edges(costs)
    rids, cids = scipy_solve(finite_costs)
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids

def lsa_solve_lapsolver(costs):
    """Solves the MWM problem using the lapsolver library."""
    from lapsolver import solve_dense

    costs = _as_dense(costs)
    rids, cids = solve_dense(costs)
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids

def lsa_solve_munkres(costs):
    """Solves the LSA problem using the Munkres library."""
    from munkres import Munkres, DISALLOWED
    m = Munkres()
    costs = _as_dense(costs)
    if not np.size(costs):
        return [], []
    # The munkres package may hang if the problem is not feasible.
    # Therefore, add expensive edges instead of using munkres.DISALLOWED.
    finite_costs = add_expensive_edges(costs)
    indices = np.array(m.compute(finite_costs), dtype=np.int)
    rids, cids = indices[:, 0], indices[:, 1]
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids

def lsa_solve_ortools(costs):
    """Solves the LSA problem using Google's optimization tools.

    Args:
        costs: Will be converted to SparseGraph if np.ndarray.
    """
    from ortools.graph import pywrapgraph

    costs = _as_sparse(costs)
    cost_values = np.array(list(costs.elems.values()))
    scale = find_scale_for_integer_approximation(cost_values)

    assignment = pywrapgraph.LinearSumAssignment()
    for (r, c), cost in costs.elems.items():
        # OR-Tools does not like to receive types like np.int64.
        if isinstance(r, np.generic):
            r = r.item()
        if isinstance(c, np.generic):
            c = c.item()
        int_cost = np.round(scale * cost).astype(int).item()
        assignment.AddArcWithCost(r, c, int_cost)

    status = assignment.Solve()
    _ortools_assert_is_optimal(pywrapgraph, status)
    return _ortools_extract_solution(assignment)

def find_scale_for_integer_approximation(costs, base=10, log_max_scale=8, log_safety=2):
    """Returns a multiplicative factor to use before rounding to integers.

    Tries to find scale = base ** j (for j integer) such that:
        abs(diff(unique(costs))) <= 1 / (scale * safety)
    where safety = base ** log_safety.

    Logs a warning if the desired resolution could not be achieved.
    """
    costs = np.asarray(costs)
    costs = costs[np.isfinite(costs)]  # Exclude non-edges (nan, inf) and -inf.
    if np.size(costs) == 0:
        # No edges with numeric value. Scale does not matter.
        return 1
    unique = np.unique(costs)
    if np.size(unique) == 1:
        # All costs have equal values. Scale does not matter.
        return 1
    try:
        _assert_integer(costs)
    except AssertionError:
        pass
    else:
        # The costs are already integers.
        return 1

    # TODO: Suppress this warning if the approximation is exact?
    # That is, if np.round(scale * costs) == scale * costs.
    warnings.warn('costs are not integers; using approximation')
    # Find scale = base ** e such that:
    # 1 / scale <= tol, or
    # e = log(scale) >= -log(tol)
    # where tol = min(diff(unique(costs)))
    min_diff = np.diff(unique).min()
    e = np.ceil(np.log(min_diff) / np.log(base)).astype(int).item()
    # Add optional non-negative safety factor to reduce quantization noise.
    e += max(log_safety, 0)
    # Ensure that we do not reduce the magnitude of the costs.
    e = max(e, 0)
    # Ensure that the scale is not too large.
    if e > log_max_scale:
        warnings.warn('could not achieve desired resolution for approximation: '
                      'want exponent %d but max is %d', e, log_max_scale)
        e = log_max_scale
    scale = base ** e
    # TODO: Check that costs * scale does not cause overflow.
    return scale

def _assert_integer(costs):
    # Check that costs are not changed by rounding.
    # Note: Elements of cost matrix may be nan, inf, -inf.
    np.testing.assert_equal(np.round(costs), costs)

def _ortools_assert_is_optimal(pywrapgraph, status):
    if status == pywrapgraph.LinearSumAssignment.OPTIMAL:
        pass
    elif status == pywrapgraph.LinearSumAssignment.INFEASIBLE:
        raise AssertionError('ortools: infeasible assignment problem')
    elif status == pywrapgraph.LinearSumAssignment.POSSIBLE_OVERFLOW:
        raise AssertionError('ortools: possible overflow in assignment problem')
    else:
        raise AssertionError('ortools: unknown status')

def _ortools_extract_solution(assignment):
    if assignment.NumNodes() == 0:
        return [], []

    pairings = []
    for i in range(assignment.NumNodes()):
        pairings.append([i, assignment.RightMate(i)])

    indices = np.array(pairings, dtype=np.int)
    return indices[:, 0], indices[:, 1]

def _assert_solution_is_feasible(costs, rids, cids):
    # Note: The matrix costs may be sparse or dense.
    ijs = list(zip(rids, cids))
    if len(ijs) != min(costs.shape):
        raise AssertionError('infeasible solution: not enough edges')
    elems = [costs[i, j] for i, j in ijs]
    if not np.all(_cost_is_edge(elems)):
        raise AssertionError('infeasible solution: includes non-finite edges')

def lsa_solve_lapjv(costs):
    from lap import lapjv

    costs = _as_dense(costs)
    if not np.size(costs):
        return [], []
    finite_costs = add_expensive_edges(costs)
    row_to_col, _ = lapjv(finite_costs, return_cost=False, extend_cost=True)
    indices = np.array([np.arange(costs.shape[0]), row_to_col], dtype=np.int).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]
    rids, cids = indices[:, 0], indices[:, 1]
    # Ensure that no missing edges were chosen.
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids

def lsa_solve_lapmod(costs):
    from lap import lapmod

    costs = _as_sparse(costs)
    if len(costs.elems) == 0:
        return [], []
    num_rows, cc, ii, kk = _sparse_to_lapmod(costs)
    # Ensure that costs are non-negative.
    cc = cc + max(0, -cc.min())
    _, row_to_col, _ = lapmod(num_rows, cc, ii, kk)

    indices = np.array([np.arange(costs.shape[0]), row_to_col], dtype=np.int).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]
    rids, cids = indices[:, 0], indices[:, 1]
    # Ensure that no missing edges were chosen.
    _assert_solution_is_feasible(costs, rids, cids)
    return rids, cids

def _sparse_to_lapmod(costs):
    num_rows, num_cols = costs.shape
    by_row = {i: {} for i in range(num_rows)}
    for (i, j), v in costs.elems.items():
        by_row[i][j] = v
    num_per_row = [len(by_row[i]) for i in range(num_rows)]
    ii = np.cumsum([0] + num_per_row)

    cc = []
    kk = []
    for i in range(num_rows):
        for j, v in sorted(by_row[i].items()):
            cc.append(v)
            kk.append(j)
    cc = np.array(cc)
    kk = np.array(kk, np.int)
    return num_rows, cc, ii, kk

def mwm_solve_greedy(costs):
    """Solves the Minimum-Weight Matching problem using a greedy approach.

    Note that edges with non-negative costs will not be used because they do not
    improve the overall cost.
    """
    costs = _as_sparse(costs)

    # Take the set of negative edges and order (ascending) by cost.
    edges = [(cost, i, j) for (i, j), cost in costs.elems.items() if cost < 0]
    edges.sort()

    # Greedily take edges with a limit of one per row and column.
    # TODO: Add warning when there are multiple edges with same cost?
    matches = []
    visited_rows = set()
    visited_cols = set()
    for _, i, j in edges:
        if i not in visited_rows and j not in visited_cols:
            matches.append((i, j))
            visited_rows.add(i)
            visited_cols.add(j)

    # Other functions return matches ordered by rows.
    matches.sort()
    rids, cids = zip(*matches) if matches else ([], [])
    return rids, cids

def init_standard_solvers():
    import importlib
    from importlib import util

    global available_solvers, default_solver, solver_map

    solvers = [
        ('lapmod', Solver(lsa_solve_lapmod, ASSIGN, module='lap')),
        ('ortools', Solver(lsa_solve_ortools, ASSIGN, module='ortools')),
        ('lap', Solver(lsa_solve_lapjv, UNBAL, module='lap')),
        ('lapsolver', Solver(lsa_solve_lapsolver, UNBAL, module='lapsolver')),
        ('scipy', Solver(lsa_solve_scipy, UNBAL, module='scipy')),
        ('munkres', Solver(lsa_solve_munkres, ASSIGN, module='munkres')),
        ('greedy', Solver(mwm_solve_greedy, MIN_WEIGHT)),
    ]

    solver_map = dict(solvers)

    available_solvers = [
            name for name, solver in solvers
            if (solver.module is None or importlib.util.find_spec(solver.module) is not None)
    ]
    if len(available_solvers) == 0:
        import warnings
        default_solver = None
        warnings.warn('No standard LAP solvers found. Consider `pip install lapsolver` or `pip install scipy`', category=RuntimeWarning)
    else:
        default_solver = available_solvers[0]

init_standard_solvers()

def _get_solver(solver):
    solver = solver or default_solver
    if isinstance(solver, six.string_types):
        solver = solver_map[solver]
    return solver

@contextmanager
def set_default_solver(newsolver):
    '''Change the default solver within context.

    Intended usage

        costs = ...
        mysolver = lambda x: ... # solver code that returns pairings

        with lap.set_default_solver(mysolver): 
            rids, cids = lap.linear_sum_assignment(costs)

    Params
    ------
    newsolver : callable or str
        new solver function
    '''

    global default_solver

    oldsolver = default_solver
    try:
        default_solver = newsolver    
        yield
    finally:
        default_solver = oldsolver

class SparseGraph(object):
    """Sparse adjacency matrix for a weighted graph.

    Missing elements have weight nan or inf, not zero.
    Mimics np.ndarray with x.shape, x[i, j] and x.transpose().
    """

    def __init__(self, shape, elems):
        self.shape = shape
        self.elems = elems

    def __getitem__(self, ij):
        return self.elems[ij]

    def __setitem__(self, ij, v):
        self.elems[ij] = v

    def transpose(self):
        m, n = self.shape
        shape = (n, m)
        elems = {(j, i): v for (i, j), v in self.elems.items()}
        return SparseGraph(shape, elems)

def sparse2dense(sparse):
    dense = np.full(sparse.shape, np.nan)
    for (i, j), v in sparse.elems.items():
        dense[i, j] = v
    return dense

def dense2sparse(dense):
    num_rows, num_cols = dense.shape
    is_edge = _cost_is_edge(dense)
    elems = {}
    for i in range(num_rows):
        for j in range(num_cols):
            if is_edge[i, j]:
                elems[i, j] = dense[i, j]
    return SparseGraph(dense.shape, elems)

def _as_sparse(costs):
    if isinstance(costs, SparseGraph):
        return costs
    elif isinstance(costs, np.ndarray):
        return dense2sparse(costs)
    else:
        raise ValueError('unknown matrix type', type(costs))

def _as_dense(costs):
    if isinstance(costs, np.ndarray):
        return costs
    elif isinstance(costs, SparseGraph):
        return sparse2dense(costs)
    else:
        raise ValueError('unknown matrix type', type(costs))
