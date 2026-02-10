#!/usr/bin/env python3
"""
Example demonstrating the new modular sim_mld structure.

This shows how to use the refactored modules with clean imports
and type-safe configuration.
"""

import sys
sys.path.insert(0, '..')

print("=" * 70)
print("Example: Using the New Modular sim_mld Structure")
print("=" * 70)

# Example 1: Type-safe Configuration
print("\n1. Type-safe Configuration")
print("-" * 70)

from sim_mld.config import (
    SimulationConfig,
    NetworkConfig,
    SolverConfig,
    ConstellationType
)

# Use pre-defined profile
config_starlink = SimulationConfig.from_profile("starlink")
print(f"Starlink profile:")
print(f"  Constellation: {config_starlink.constellation_type.value}")
print(f"  LCTs per sat: {config_starlink.network.n_lct_per_sat}")
print(f"  Max ISL distance: {config_starlink.network.max_lisl_distance_km} km")

# Custom configuration
config_custom = SimulationConfig(
    constellation_type=ConstellationType.CUSTOM,
    network=NetworkConfig(
        n_lct_per_sat=2,
        max_lisl_distance_km=2000.0
    ),
    solver=SolverConfig(
        alpha=0.05,
        beta=0.5
    ),
    visualization=None  # Headless
)
print(f"\nCustom configuration:")
print(f"  LCTs per sat: {config_custom.network.n_lct_per_sat}")
print(f"  Solver alpha: {config_custom.solver.alpha}")
print(f"  Visualization: {'Enabled' if config_custom.visualization else 'Headless'}")

# Example 2: Configuration Validation
print("\n2. Configuration Validation")
print("-" * 70)

try:
    # This will fail validation
    bad_config = NetworkConfig(n_lct_per_sat=-1)
except ValueError as e:
    print(f"✓ Validation caught error: {e}")

try:
    # This will also fail
    bad_config = NetworkConfig(max_lisl_distance_km=-100)
except ValueError as e:
    print(f"✓ Validation caught error: {e}")

# Example 3: Available Profiles
print("\n3. Available Configuration Profiles")
print("-" * 70)

profiles = ["starlink", "kuiper", "oneweb", "minimal"]
for profile_name in profiles:
    try:
        profile = SimulationConfig.from_profile(profile_name)
        print(f"✓ {profile_name:10s}: {profile.constellation_type.value:10s} " +
              f"({profile.network.n_lct_per_sat} LCTs/sat)")
    except ValueError as e:
        print(f"✗ {profile_name}: {e}")

# Example 4: Enums for Type Safety
print("\n4. Enum-based Type Safety")
print("-" * 70)

from sim_mld.config import CapacityModelType, MatchingStrategyType, RoutingStrategyType

print(f"Capacity models: {[e.value for e in CapacityModelType]}")
print(f"Matching strategies: {[e.value for e in MatchingStrategyType]}")
print(f"Routing strategies: {[e.value for e in RoutingStrategyType]}")

# Example 5: Backward Compatibility
print("\n5. Backward Compatibility")
print("-" * 70)
print("The new structure maintains backward compatibility.")
print("Old imports still work, new imports provide better organization.")
print()
print("Old way (still works):")
print("  from sim_mld.constants import EARTH_RADIUS_KM")
print()
print("New way (recommended):")
print("  from sim_mld.core import EARTH_RADIUS_KM")

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
The new modular structure provides:
✓ Type-safe configuration with validation
✓ Pre-defined profiles for common scenarios
✓ Enum-based type safety
✓ Backward compatibility
✓ Clear package organization
✓ Lightweight - config works without heavy dependencies
""")
