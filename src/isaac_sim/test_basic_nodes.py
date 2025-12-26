#!/usr/bin/env python3
"""
Isaac Sim Basic Nodes and APIs Validation Script

This script validates basic Isaac Sim nodes and APIs without implementing
perception, navigation, or fusion logic.
"""

import sys
import importlib.util

def check_basic_nodes():
    """Check if basic Isaac Sim nodes can be imported and used."""
    print("Checking basic Isaac Sim nodes and APIs...")

    # Test basic Isaac Sim core functionality
    try:
        from omni.isaac.core import World
        from omni.isaac.core.robots import Robot
        from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage
        from omni.isaac.core.utils.nucleus import get_assets_root_path
        from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
        from omni.isaac.core.utils.viewports import set_camera_view

        print("✓ Basic Isaac Sim core classes importable")

        # Test basic API functions
        try:
            # These are just import/syntax checks, not actual simulation execution
            create_new_stage
            get_prim_at_path
            define_prim
            set_camera_view
            print("✓ Basic Isaac Sim APIs accessible")
            return True
        except Exception as e:
            print(f"! Error accessing basic APIs: {e}")
            return False

    except ImportError as e:
        print(f"! Error importing basic Isaac Sim nodes: {e}")
        return False

def check_sensor_nodes():
    """Check if basic sensor nodes can be imported."""
    print("\nChecking basic Isaac Sim sensor nodes...")

    try:
        # Check if basic sensor functionality is available
        from omni.isaac.sensor import SensorPrim
        from omni.isaac.range_sensor import RayCaster, LidarRtx
        from omni.isaac.core.sensors import Camera, ImuSensor

        print("✓ Basic Isaac Sim sensor nodes importable")

        # Test basic sensor API functions
        try:
            # These are just import/syntax checks
            RayCaster
            LidarRtx
            Camera
            ImuSensor
            print("✓ Basic sensor APIs accessible")
            return True
        except Exception as e:
            print(f"! Error accessing sensor APIs: {e}")
            return False

    except ImportError as e:
        print(f"! Error importing sensor nodes: {e}")
        return False

def check_physics_nodes():
    """Check if basic physics nodes can be imported."""
    print("\nChecking basic Isaac Sim physics nodes...")

    try:
        # Check if basic physics functionality is available
        from omni.isaac.core.physics import PhysicsSchema
        from omni.isaac.core.objects import DynamicCuboid, FixedCuboid
        from omni.isaac.core.prims import RigidPrim, Articulation, XFormPrim

        print("✓ Basic Isaac Sim physics nodes importable")

        # Test basic physics API functions
        try:
            # These are just import/syntax checks
            DynamicCuboid
            FixedCuboid
            RigidPrim
            XFormPrim
            print("✓ Basic physics APIs accessible")
            return True
        except Exception as e:
            print(f"! Error accessing physics APIs: {e}")
            return False

    except ImportError as e:
        print(f"! Error importing physics nodes: {e}")
        return False

def check_world_and_stage():
    """Check if World and Stage management APIs are available."""
    print("\nChecking World and Stage management APIs...")

    try:
        from omni.isaac.core import World
        from omni.isaac.core.utils.stage import add_reference_to_stage, create_new_stage, get_stage_units
        from omni.isaac.core.utils.bounds import compute_aabb, create_bbox_cache
        from omni.isaac.core.utils.semantics import add_semantics

        print("✓ World and Stage management APIs accessible")

        # Test basic world functions
        try:
            # These are just import/syntax checks
            World
            create_new_stage
            get_stage_units
            print("✓ World management functions accessible")
            return True
        except Exception as e:
            print(f"! Error accessing world functions: {e}")
            return False

    except ImportError as e:
        print(f"! Error importing world/stage modules: {e}")
        return False

def check_utils():
    """Check if basic Isaac Sim utilities are available."""
    print("\nChecking basic Isaac Sim utilities...")

    try:
        from omni.isaac.core.utils import transformations
        from omni.isaac.core.utils.rotations import gf_quat_to_np_array, np_quat_to_gf_quat
        from omni.isaac.core.utils.carb_utils import get_current_budget, set_current_budget

        print("✓ Basic Isaac Sim utilities accessible")

        # Test basic utility functions
        try:
            # These are just import/syntax checks
            transformations
            gf_quat_to_np_array
            np_quat_to_gf_quat
            print("✓ Basic utility functions accessible")
            return True
        except Exception as e:
            print(f"! Error accessing utility functions: {e}")
            return False

    except ImportError as e:
        print(f"! Error importing utility modules: {e}")
        return False

def main():
    print("Isaac Sim Basic Nodes and APIs Validation")
    print("="*50)

    # Test 1: Basic nodes
    test1_result = check_basic_nodes()

    # Test 2: Sensor nodes
    test2_result = check_sensor_nodes()

    # Test 3: Physics nodes
    test3_result = check_physics_nodes()

    # Test 4: World and Stage APIs
    test4_result = check_world_and_stage()

    # Test 5: Utilities
    test5_result = check_utils()

    print("\n" + "="*50)
    print("Validation Summary:")
    print(f"Basic Nodes: {'PASS' if test1_result else 'FAIL'}")
    print(f"Sensor Nodes: {'PASS' if test2_result else 'FAIL'}")
    print(f"Physics Nodes: {'PASS' if test3_result else 'FAIL'}")
    print(f"World/Stage APIs: {'PASS' if test4_result else 'FAIL'}")
    print(f"Utilities: {'PASS' if test5_result else 'FAIL'}")

    all_passed = all([test1_result, test2_result, test3_result, test4_result, test5_result])

    if all_passed:
        print("\n✓ Isaac Sim basic nodes and APIs validation successful!")
        print("  All basic Isaac Sim components are accessible.")
        return 0
    else:
        print("\n✗ Isaac Sim basic nodes and APIs validation incomplete!")
        print("  Some basic components are not accessible.")
        return 1

if __name__ == "__main__":
    sys.exit(main())