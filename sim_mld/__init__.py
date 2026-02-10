"""
sim_mld: Satellite constellation network optimization and simulation.

This package provides modular components for LEO satellite network simulations:
- core: Pure mathematical functions (coordinates, constants)
- network: Link models, capacity calculations, graph operations
- optimization: Matching and routing strategies
- visualization: Pluggable rendering backends
- config: Type-safe configuration management

For backward compatibility, the old flat structure is maintained.
"""

# Only import cvxpy when actually needed (lazy import)
def _check_cvxpy():
    """Check if cvxpy is available."""
    try:
        import cvxpy as cp
        print(cp.installed_solvers())
    except ImportError:
        pass

# Don't run the check on import - only when explicitly called
# _check_cvxpy()
