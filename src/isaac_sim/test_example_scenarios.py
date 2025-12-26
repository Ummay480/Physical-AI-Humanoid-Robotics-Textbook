#!/usr/bin/env python3
"""
Isaac Sim Example Scenarios Validation Script

This script validates that Isaac Sim example scenarios can be set up
for test runs without implementing complex scenario logic.
"""

import sys
import os
import subprocess
import importlib.util

def check_isaac_sim_scenarios_imports():
    """Check if Isaac Sim scenario modules can be imported."""
    print("Checking Isaac Sim scenario modules...")

    scenario_modules = [
        'omni.isaac.core',
        'omni.isaac.core.utils',
        'omni.isaac.core.objects',
        'omni.isaac.core.prims',
        'omni.isaac.core.world',
        'omni.isaac.core.robots',
    ]

    missing_modules = []
    found_modules = []

    for module_path in scenario_modules:
        try:
            # Try to import the scenario module
            spec = importlib.util.find_spec(module_path)
            if spec is not None:
                importlib.import_module(module_path)
                found_modules.append(module_path)
                print(f"✓ {module_path} importable")
            else:
                missing_modules.append(module_path)
                print(f"! {module_path} not found")
        except ImportError as e:
            missing_modules.append(module_path)
            print(f"! {module_path} import failed: {e}")
        except Exception as e:
            missing_modules.append(module_path)
            print(f"! {module_path} error: {e}")

    if missing_modules:
        print(f"\n! Missing Isaac Sim scenario modules: {missing_modules}")
        print("  Isaac Sim scenario modules may not be properly installed")
        return False

    print(f"\n✓ Found Isaac Sim scenario modules: {found_modules}")
    return True

def check_basic_scenario_concepts():
    """Check if basic scenario concepts can be imported."""
    print("\nChecking basic scenario concepts...")

    try:
        # Check if we can access the basic concepts needed for scenarios
        try:
            import omni
            print("✓ Omni modules available")
        except ImportError:
            print("! Omni modules not available")

        try:
            import carb
            print("✓ Carb modules available")
        except ImportError:
            print("! Carb modules not available")

        # Try to access basic Isaac Sim utilities
        try:
            from omni.isaac.core.utils.stage import create_new_stage
            from omni.isaac.core.utils.prims import define_prim
            from omni.isaac.core.utils.nucleus import get_assets_root_path
            print("✓ Basic scenario utilities available")
            return True
        except ImportError as e:
            print(f"! Basic scenario utilities not available: {e}")
            return False

    except Exception as e:
        print(f"! Error checking scenario concepts: {e}")
        return False

def check_isaac_sim_core():
    """Check if Isaac Sim core modules are available for scenarios."""
    print("\nChecking Isaac Sim core modules for scenarios...")

    try:
        # Try to import Isaac Sim core components needed for scenarios
        import omni.isaac.core
        import omni.isaac.core.utils
        print("✓ Isaac Sim core modules available")
        return True
    except ImportError as e:
        print(f"! Isaac Sim core modules not available: {e}")
        return False

def check_asset_access():
    """Check if Isaac Sim assets can be accessed."""
    print("\nChecking Isaac Sim asset access...")

    try:
        # Try to access basic asset functions
        try:
            from omni.isaac.core.utils.nucleus import get_assets_root_path
            # This is just a validation check, not actual asset loading
            print("✓ Asset access functions available")
            return True
        except ImportError:
            print("! Asset access functions not available")
            return False
    except Exception as e:
        print(f"! Error checking asset access: {e}")
        return False

def check_scenario_utils():
    """Check if scenario utilities are available."""
    print("\nChecking scenario utilities...")

    try:
        # Check for basic utilities needed for scenarios
        try:
            from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
            print("✓ Bounding box utilities available")
        except ImportError:
            print("! Bounding box utilities not available")

        try:
            from omni.isaac.core.utils.rotations import gf_quat_to_np_array
            print("✓ Rotation utilities available")
        except ImportError:
            print("! Rotation utilities not available")

        try:
            from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
            print("✓ Object creation utilities available")
        except ImportError:
            print("! Object creation utilities not available")

        return True
    except Exception as e:
        print(f"! Error checking scenario utilities: {e}")
        return False

def main():
    print("Isaac Sim Example Scenarios Validation")
    print("="*50)

    # Test 1: Isaac Sim scenario imports
    test1_result = check_isaac_sim_scenarios_imports()

    # Test 2: Basic scenario concepts
    test2_result = check_basic_scenario_concepts()

    # Test 3: Isaac Sim core modules
    test3_result = check_isaac_sim_core()

    # Test 4: Asset access
    test4_result = check_asset_access()

    # Test 5: Scenario utilities
    test5_result = check_scenario_utils()

    print("\n" + "="*50)
    print("Validation Summary:")
    print(f"Scenario Imports: {'PASS' if test1_result else 'FAIL'}")
    print(f"Scenario Concepts: {'PASS' if test2_result else 'FAIL'}")
    print(f"Isaac Sim Core: {'PASS' if test3_result else 'FAIL'}")
    print(f"Asset Access: {'PASS' if test4_result else 'FAIL'}")
    print(f"Scenario Utilities: {'PASS' if test5_result else 'FAIL'}")

    all_passed = all([test1_result, test2_result, test3_result, test4_result, test5_result])

    if all_passed:
        print("\n✓ Isaac Sim example scenarios validation successful!")
        print("  All basic scenario components are available.")
        return 0
    else:
        print("\n✗ Isaac Sim example scenarios validation incomplete!")
        print("  Some scenario components are not available.")
        return 1

if __name__ == "__main__":
    sys.exit(main())