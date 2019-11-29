import numpy as np
from collections import OrderedDict
import logging
import warnings


# Largest magnitude integer that float32 can represent exactly.
# https://stackoverflow.com/questions/3793838/
# To see this:
# >>> 2 ** 24 + 1
# 16777217
# >>> np.float32(2 ** 24 + 1)
# 16777216.0
# >>> np.float32(2 ** 24 - 1)
# 16777215.0
FLOAT32_MAX_INT = 2 ** 24 - 1

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

    solver = solver or default_solver

    if isinstance(solver, str):
        # Try resolve from string
        solver = solver_map.get(solver, None)    
    
    assert callable(solver), 'Invalid LAP solver.'
    rids, cids = solver(costs)
    rids = np.asarray(rids).astype(int)
    cids = np.asarray(cids).astype(int)
    return rids, cids

def _replace_nan_with_large_constant(costs):
    # The linear_sum_assignment function in scipy does not support missing edges.
    # Replace nan with a large constant that ensures it is not chosen.
    # If it is chosen, that means the problem was infeasible.
    valid = np.isfinite(costs)
    if valid.all():
        return costs
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

    finite_costs = _replace_nan_with_large_constant(costs)
    rids, cids = scipy_solve(finite_costs)

    # Ensure that no missing edges were chosen.
    matching_costs = costs[rids, cids]
    if not np.all(np.isfinite(matching_costs)):
        raise ValueError('infeasible problem')
    return rids, cids

def lsa_solve_lapsolver(costs):
    """Solves the LSA problem using the lapsolver library."""
    from lapsolver import solve_dense
    return solve_dense(costs)

def lsa_solve_munkres(costs):
    """Solves the LSA problem using the Munkres library."""
    from munkres import Munkres, DISALLOWED
    m = Munkres()

    costs = costs.copy()
    inv = ~np.isfinite(costs)
    if inv.any():
        costs = costs.astype(object)
        costs[inv] = DISALLOWED       

    indices = np.array(m.compute(costs), dtype=np.int64)
    return indices[:,0], indices[:,1]

def _assert_integer(costs):
    # Check that costs are not changed by rounding.
    # Note: Elements of cost matrix may be nan.
    np.testing.assert_equal(np.round(costs), costs)
    # Require that costs are within representable range of integers for float32.
    # This guarantees that int() will work.
    # Note: Elements of cost matrix may be nan.
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        within_range = np.logical_not(np.abs(costs) > FLOAT32_MAX_INT)
    if not np.all(within_range):
        raise AssertionError('costs are too large', np.max(np.abs(costs)))

def lsa_solve_ortools(costs):
    """Solves the LSA problem using Google's optimization tools."""
    from ortools.graph import pywrapgraph

    # Google OR tools only support integer costs. Here's our attempt
    # to convert from floating point to integer:
    #
    # We search for the minimum difference between any two costs and
    # compute the first non-zero digit after the decimal place. Then
    # we compute a factor,f, that scales all costs so that the difference
    # is integer representable in the first digit.
    # 
    # Example: min-diff is 0.001, then first non-zero digit place -3, so
    # we scale by 1e3.
    #
    # For small min-diffs and large costs in general there is a change of
    # overflowing.

    log10_scale = find_scale_for_integer_approximation(costs)
    scale = 10 ** log10_scale

    valid = np.isfinite(costs)
    assignment = pywrapgraph.LinearSumAssignment()
    for r in range(costs.shape[0]):
        for c in range(costs.shape[1]):
            if valid[r,c]:
                assignment.AddArcWithCost(r, c, int(np.round(costs[r, c] * scale)))

    status = assignment.Solve()
    _ortools_assert_is_optimal(pywrapgraph, status)
    return _ortools_extract_solution(assignment)

def find_scale_for_integer_approximation(costs, max_log10_scale=8, log10_safety=2):
    costs = np.asarray(costs)
    costs = costs[np.isfinite(costs)]
    if np.size(costs) == 0:
        return 0

    try:
        _assert_integer(costs)
    except AssertionError:
        pass
    else:
        return 0
    logging.warning('costs are not integers; using approximation')

    # Find scale = 10 ** e such that:
    # 1 / scale <= tol, or
    # e = log(scale) >= -log tol
    # where tol = min(diff(unique(costs)))
    unique = np.unique(costs)
    if np.size(unique) == 1:
        # The magnitude of the cost does not matter at all.
        return 0
    min_diff = np.diff(unique).min()
    e = int(np.ceil(np.log10(min_diff)))
    # Add optional non-negative safety factor to reduce quantization noise.
    e += max(log10_safety, 0)
    # Ensure that we do not reduce the magnitude of the costs.
    e = max(e, 0)
    # Ensure that the scale is not too large.
    if e > max_log10_scale:
        logging.warning('could not achieve desired resolution for integer approximation: '
                        'want 10 ** %d but max is 10 ** %d', e, max_log10_scale)
        e = max_log10_scale
    # TODO(valmadre): Check that costs * 10 ** e does not cause overflow.
    return e

def _ortools_assert_is_optimal(pywrapgraph, status):
    if status == pywrapgraph.LinearSumAssignment.OPTIMAL:
        pass
    elif status == pywrapgraph.LinearSumAssignment.INFEASIBLE:
        raise ValueError('infeasible assignment problem')
    elif status == pywrapgraph.LinearSumAssignment.POSSIBLE_OVERFLOW:
        raise ValueError('possible overflow in assignment problem')
    else:
        raise ValueError('unknown status')

def _ortools_extract_solution(assignment):
    if assignment.NumNodes() == 0:
        return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

    pairings = []
    for i in range(assignment.NumNodes()):
        pairings.append([i, assignment.RightMate(i)])

    indices = np.array(pairings, dtype=np.int64)
    return indices[:, 0], indices[:, 1]

def _assert_solution_is_valid(costs, rids, cids):
    matching_costs = costs[rids, cids]
    if not np.all(np.isfinite(matching_costs)):
        raise AssertionError('infeasible problem')

def lsa_solve_lapjv(costs):
    from lap import lapjv

    finite_costs = _replace_nan_with_large_constant(costs)
    row_to_col, _ = lapjv(finite_costs, return_cost=False, extend_cost=True)
    indices = np.array((range(costs.shape[0]), row_to_col), dtype=np.int64).T
    # Exclude unmatched rows (in case of unbalanced problem).
    indices = indices[indices[:, 1] != -1]
    rids, cids = indices[:,0], indices[:,1]
    # Ensure that no missing edges were chosen.
    _assert_solution_is_valid(costs, rids, cids)
    return rids, cids

def init_standard_solvers():
    import importlib
    from importlib import util
    
    global available_solvers, default_solver, solver_map

    solvers = [
        ('lapsolver', lsa_solve_lapsolver),
        ('lap', lsa_solve_lapjv),        
        ('scipy', lsa_solve_scipy),
        ('munkres', lsa_solve_munkres),
        ('ortools', lsa_solve_ortools),
    ]

    solver_map = dict(solvers)    
    
    available_solvers = [s[0] for s in solvers if importlib.util.find_spec(s[0]) is not None]
    if len(available_solvers) == 0:
        import warnings
        default_solver = None        
        warnings.warn('No standard LAP solvers found. Consider `pip install lapsolver` or `pip install scipy`', category=RuntimeWarning)
    else:
        default_solver = available_solvers[0]

init_standard_solvers()

from contextlib import contextmanager

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

