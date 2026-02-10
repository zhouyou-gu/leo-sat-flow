# Using the New Modular sim_mld Structure

This guide shows how to use the refactored sim_mld module with its new modular architecture.

## Quick Start

### Type-Safe Configuration

The new configuration system uses dataclasses with validation:

```python
from sim_mld.config import SimulationConfig, NetworkConfig

# Use pre-defined profile
config = SimulationConfig.from_profile("starlink")
print(config.network.n_lct_per_sat)  # 4

# Custom configuration with validation
config = SimulationConfig(
    constellation_type="custom",
    network=NetworkConfig(
        n_lct_per_sat=2,
        max_lisl_distance_km=2000
    ),
    visualization=None  # Headless mode
)
```

### Available Profiles

Pre-configured profiles for common scenarios:

- **starlink**: 4 LCTs/sat, 5000 km max ISL distance
- **kuiper**: 4 LCTs/sat, 4000 km max ISL distance
- **oneweb**: 2 LCTs/sat, 3000 km max ISL distance
- **minimal**: 2 LCTs/sat, 2000 km max ISL distance (headless)

```python
config = SimulationConfig.from_profile("starlink")
```

## New Package Structure

```
sim_mld/
├── core/           # Pure mathematical functions
│   ├── coordinates.py   # Coordinate transformations
│   └── constants.py     # Physical constants
├── network/        # Link models and topology
│   ├── interfaces.py    # Abstract base classes
│   ├── capacity.py      # Capacity calculators
│   ├── channel_model.py # Channel physics
│   └── graph.py         # Graph operations
├── optimization/   # Matching and routing strategies
│   ├── matching.py      # Link matching
│   └── routing.py       # Path routing
├── visualization/  # Pluggable rendering backends
│   └── interfaces.py    # Visualization ABC
└── config/         # Configuration management
    └── schema.py        # Type-safe configs
```

## Import Examples

### New Modular Imports (Recommended)

```python
# Configuration
from sim_mld.config import SimulationConfig, NetworkConfig

# Core functions (when numba available)
from sim_mld.core import lat_lon_to_xyz, EARTH_RADIUS_KM

# Network models (when dependencies available)
from sim_mld.network import LinkCapacityModel, OpticalGaussianLinkModel

# Optimization strategies
from sim_mld.optimization import GreedyMatchingStrategy, DijkstraRoutingStrategy

# Visualization interfaces
from sim_mld.visualization import VisualizationBackend, NullVisualizationBackend
```

### Old Imports (Still Work)

Backward compatibility maintained:

```python
from sim_mld.constants import EARTH_RADIUS_KM
from sim_mld.capacity_calculator import CapacityCalculator
from sim_mld.matching_strategy import GreedyMatchingStrategy
```

## Configuration Validation

All configuration parameters are validated:

```python
from sim_mld.config import NetworkConfig

try:
    config = NetworkConfig(n_lct_per_sat=-1)  # Invalid!
except ValueError as e:
    print(e)  # "n_lct_per_sat must be >= 1, got -1"
```

## Type-Safe Enums

Use enums for algorithm selection:

```python
from sim_mld.config import (
    CapacityModelType,
    MatchingStrategyType,
    RoutingStrategyType
)

# Available capacity models
print(CapacityModelType.OPTICAL_GAUSSIAN)  # "optical_gaussian"
print(CapacityModelType.RF_LINK_BUDGET)    # "rf_link_budget"

# Available matching strategies
print(MatchingStrategyType.GREEDY)      # "greedy"
print(MatchingStrategyType.HUNGARIAN)   # "hungarian"

# Available routing strategies
print(RoutingStrategyType.DIJKSTRA)  # "dijkstra"
print(RoutingStrategyType.OSPF)      # "ospf"
```

## Lazy Loading & Dependencies

The new structure uses intelligent fallbacks:

- **Config system**: Works without numba/cvxpy (lightweight)
- **Core module**: Requires numba for coordinate functions
- **Network module**: Requires numpy, optionally uses numba
- **Optimization**: Falls back to old imports if new structure unavailable

All modules use try/except for graceful degradation:

```python
try:
    from sim_mld.core.constants import EARTH_RADIUS_KM
except ImportError:
    from sim_mld.constants import EARTH_RADIUS_KM  # Fallback
```

## Benefits

1. **✓ Self-contained**: Core has zero external dependencies
2. **✓ Extensible**: Strategy pattern enables easy algorithm addition
3. **✓ Understandable**: Clear package hierarchy
4. **✓ Type-safe**: Dataclass configs with validation
5. **✓ Backward compatible**: Old imports still work
6. **✓ Lightweight**: Config works independently

## Testing

Run the configuration tests:

```bash
python << 'EOF'
from sim_mld.config import SimulationConfig
config = SimulationConfig.from_profile("starlink")
print(f"✓ Config works: {config.constellation_type.value}")
EOF
```

## Next Steps

- See `SIM_MLD_REFACTORING_GUIDE.md` for complete architecture
- See `SIM_MLD_COMPLETE_REFACTORING.md` for implementation details
- Check examples in `examples/` directory

## Migration Guide

For existing code:

1. **Keep using old imports** - they still work
2. **Gradually adopt new imports** - cleaner and more organized
3. **Use config system** - type-safe parameter management
4. **Add new algorithms** - via strategy pattern, no code modification needed

The refactoring maintains 100% backward compatibility while providing a better foundation for future development.
