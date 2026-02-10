# Complete sim_mld Refactoring Guide

## Overview

This document describes the complete refactoring of the `sim_mld` module to create a **self-contained**, **extensible**, and **understandable** architecture for LEO satellite network simulations.

## Goals

1. **Self-contained**: Minimal external dependencies, clear module boundaries
2. **Extensible**: Easy to add new algorithms without modifying existing code
3. **Understandable**: Clear package structure, explicit dependencies, comprehensive documentation

## New Architecture

### Package Structure

```
sim_mld/
├── core/                   # Pure mathematical functions (zero external deps)
│   ├── __init__.py
│   ├── coordinates.py      # Coordinate transformations (spherical ↔ Cartesian)
│   └── constants.py        # Physical and mathematical constants
│
├── network/                # Network topology and link models
│   ├── __init__.py
│   ├── interfaces.py       # ABC for LinkCapacityModel, LinkPhysicsModel
│   ├── channel_model.py    # Optical/RF channel physics
│   ├── capacity.py         # Capacity calculator implementations
│   └── graph.py            # Graph operations (CSR, Dijkstra)
│
├── optimization/           # Network optimization algorithms
│   ├── __init__.py
│   ├── matching.py         # Link matching strategies (greedy, Hungarian)
│   ├── routing.py          # Path routing strategies (Dijkstra, OSPF)
│   └── solver.py           # Coordinating solver (future)
│
├── simulation/             # Simulation orchestration (future)
│   ├── __init__.py
│   ├── scenario.py         # Simulation scenario management
│   ├── state.py            # Constellation state
│   └── coordinator.py      # Orchestrates network + simulation steps
│
├── visualization/          # Pluggable visualization backends
│   ├── __init__.py
│   ├── interfaces.py       # ABC for VisualizationBackend
│   └── vispy_backend.py    # Vispy implementation (future)
│
├── config/                 # Configuration management
│   ├── __init__.py
│   └── schema.py           # Dataclass-based config with validation
│
├── ml/                     # Machine learning models (existing)
│   ├── __init__.py
│   ├── base_model.py
│   ├── ld_sgl/
│   ├── pd_sgl/
│   └── e2e_rl/
│
└── tests/                  # Test infrastructure
    ├── __init__.py
    ├── unit/               # Unit tests
    ├── integration/        # Integration tests
    └── fixtures/           # Test fixtures and mocks
```

### Backward Compatibility

To maintain compatibility with existing code, the old flat structure is preserved:
- `sim_mld/constellation.py` → Can import from `sim_mld.core.coordinates`
- `sim_mld/constants.py` → Aliases to `sim_mld.core.constants`
- `sim_mld/capacity_calculator.py` → Can import from `sim_mld.network.capacity`
- etc.

## Design Patterns Applied

### 1. Strategy Pattern

**Purpose**: Enable runtime algorithm selection without changing client code.

**Implementations**:
- `MatchingStrategy` hierarchy (greedy, Hungarian, network flow)
- `RoutingStrategy` hierarchy (Dijkstra, OSPF, load-balanced)
- `LinkCapacityModel` hierarchy (optical Gaussian, RF, custom)
- `VisualizationBackend` hierarchy (Vispy, Matplotlib, null)

**Example**:
```python
from sim_mld.optimization import GreedyMatchingStrategy, DijkstraRoutingStrategy
from sim_mld.config import SolverConfig

config = SolverConfig(
    matching_strategy="greedy",
    routing_strategy="dijkstra"
)

# Strategies are injected via dependency injection
solver = NetworkSolver(
    matching=GreedyMatchingStrategy(),
    routing=DijkstraRoutingStrategy()
)
```

### 2. Null Object Pattern

**Purpose**: Provide do-nothing implementations to avoid conditional logic.

**Implementation**: `NullVisualizationBackend` for headless simulations

**Example**:
```python
# Headless mode - no visualization overhead
config = SimulationConfig(visualization=None)
simulation = Simulation(config)
simulation.run()  # No visualization checks needed
```

### 3. Factory Pattern

**Purpose**: Create complex objects with pre-configured settings.

**Implementation**: `SimulationConfig.from_profile()`

**Example**:
```python
# Quick setup with pre-defined profiles
config = SimulationConfig.from_profile("starlink")
config = SimulationConfig.from_profile("kuiper")
config = SimulationConfig.from_profile("minimal")  # Headless, fast
```

### 4. Dependency Injection

**Purpose**: Decouple components for testability and flexibility.

**Example**:
```python
class NetworkSolver:
    def __init__(
        self,
        matching_strategy: MatchingStrategy,
        routing_strategy: RoutingStrategy,
        capacity_model: LinkCapacityModel
    ):
        self.matching = matching_strategy
        self.routing = routing_strategy
        self.capacity = capacity_model
```

## Key Abstractions

### LinkCapacityModel Interface

```python
class LinkCapacityModel(ABC):
    @abstractmethod
    def compute_capacity(self, distance: np.ndarray, **kwargs) -> np.ndarray:
        """Compute link capacity from distance."""
        pass
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """Get model parameters."""
        pass
```

**Implementations**:
- `OpticalGaussianLinkModel` - Current implementation
- `RFLinkBudgetModel` - Future RF satellites
- `MLBasedCapacityModel` - Future ML-predicted capacity

### VisualizationBackend Interface

```python
class VisualizationBackend(ABC):
    @abstractmethod
    def update_satellite_positions(self, positions: np.ndarray, colors: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def update_links(self, link_positions: np.ndarray, link_colors: np.ndarray) -> None:
        pass
    
    @abstractmethod
    def render(self) -> None:
        pass
```

**Implementations**:
- `NullVisualizationBackend` - Headless (no-op)
- `VispyVisualizationBackend` - Current 3D renderer
- `MatplotlibVisualizationBackend` - Future 2D plots

### Configuration Schema

```python
@dataclass
class SimulationConfig:
    constellation_type: ConstellationType
    network: NetworkConfig
    solver: SolverConfig
    visualization: Optional[VisualizationConfig]
    
    def __post_init__(self):
        """Validate all parameters."""
        # Automatic validation
```

## Migration Guide

### For Users

**Old way (still works)**:
```python
from sim_mld.constellation import lat_lon_to_xyz
from sim_mld.constants import EARTH_RADIUS_KM
from sim_mld.capacity_calculator import CapacityCalculator
```

**New way (recommended)**:
```python
from sim_mld.core import lat_lon_to_xyz, EARTH_RADIUS_KM
from sim_mld.network import OpticalGaussianLinkModel
from sim_mld.config import SimulationConfig

# Type-safe configuration
config = SimulationConfig.from_profile("starlink")
model = OpticalGaussianLinkModel()
capacity = model.compute_capacity(distance)
```

### For Developers

**Adding a new matching algorithm**:

1. Create new strategy class:
```python
# sim_mld/optimization/matching.py
class HungarianMatchingStrategy(MatchingStrategy):
    def compute_matching(self, weighted_edges):
        # Your implementation
        return matched_pairs
```

2. Register in enum:
```python
# sim_mld/config/schema.py
class MatchingStrategyType(str, Enum):
    GREEDY = "greedy"
    HUNGARIAN = "hungarian"  # Add new type
```

3. Use it:
```python
config = SolverConfig(matching_strategy="hungarian")
```

**Adding a new visualization backend**:

1. Implement interface:
```python
# sim_mld/visualization/matplotlib_backend.py
class MatplotlibVisualizationBackend(VisualizationBackend):
    def update_satellite_positions(self, positions, colors):
        # Matplotlib implementation
        pass
```

2. Use it:
```python
from sim_mld.visualization import MatplotlibVisualizationBackend
backend = MatplotlibVisualizationBackend()
```

## Testing

### Unit Tests

```python
# sim_mld/tests/unit/test_coordinates.py
def test_lat_lon_to_xyz():
    from sim_mld.core import lat_lon_to_xyz
    import numpy as np
    
    lat_lon = np.array([[0, 0]])  # Equator, prime meridian
    xyz = lat_lon_to_xyz(lat_lon, r=1.0)
    
    assert np.allclose(xyz, [[1, 0, 0]])
```

### Integration Tests

```python
# sim_mld/tests/integration/test_solver.py
def test_solver_with_strategies():
    from sim_mld.optimization import GreedyMatchingStrategy, DijkstraRoutingStrategy
    
    solver = NetworkSolver(
        matching=GreedyMatchingStrategy(),
        routing=DijkstraRoutingStrategy()
    )
    
    result = solver.optimize(network_graph)
    assert result.objective > 0
```

## Performance Considerations

- **Numba JIT preserved**: All hot paths still use `@njit` decorators
- **Zero overhead abstractions**: Interfaces don't add runtime cost
- **Lazy initialization**: Backends created only when needed
- **Configuration validation**: Done once at startup, not in loops

## Future Extensions

### Phase 3 (Planned)

1. **Complete simulation orchestration**
   - Split `simulation.py` into modular components
   - Separate state management from visualization
   - Event-driven architecture

2. **Registry pattern for strategies**
   ```python
   StrategyRegistry.register("my_matching", MyMatchingStrategy)
   solver = StrategyRegistry.create_solver("my_matching", "dijkstra")
   ```

3. **Plugin system**
   ```python
   # User-defined plugins
   from sim_mld.plugins import register_capacity_model
   
   @register_capacity_model("ml_capacity")
   class MLCapacityModel(LinkCapacityModel):
       ...
   ```

4. **Comprehensive test coverage**
   - Unit tests for all modules
   - Integration tests for workflows
   - Performance benchmarks

## Benefits Summary

| Aspect | Before | After |
|--------|--------|-------|
| **Structure** | Flat 15-file directory | Hierarchical packages by concern |
| **Dependencies** | Star imports, hidden deps | Explicit imports, clear hierarchy |
| **Extensibility** | Modify existing code | Add new strategy classes |
| **Configuration** | Scattered constants | Type-safe dataclasses |
| **Visualization** | Hardcoded Vispy | Pluggable backends |
| **Testing** | No infrastructure | Unit/integration/fixtures |
| **Documentation** | Minimal | Comprehensive with examples |

## Conclusion

The refactored `sim_mld` module provides a solid foundation for:
- **Research**: Easy to experiment with new algorithms
- **Development**: Clear structure accelerates feature addition
- **Maintenance**: Explicit dependencies simplify debugging
- **Extension**: Plugin architecture welcomes contributions

The modular design maintains backward compatibility while enabling future growth.
