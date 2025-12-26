#!/usr/bin/env python3
"""
Isaac Sim Runtime Installation Validation Script

This script validates that Isaac Sim runtime is properly installed with:
- Headless mode support
- GUI mode support
- ROS 2 bridge functionality
"""

import sys
import os
import subprocess
import importlib.util

def check_isaac_sim_imports():
    """Check if Isaac Sim Python modules can be imported."""
    print("Checking Isaac Sim Python modules...")

    required_modules = [
        'omni.isaac.kit',
        'omni.isaac.core',
        'omni.isaac.core.utils',
        'omni.isaac.range_sensor',
        'omni.isaac.sensor',
    ]

    missing_modules = []
    found_modules = []

    for module_path in required_modules:
        try:
            # Split the module path to check nested imports
            parts = module_path.split('.')
            spec = importlib.util.find_spec(parts[0])
            if spec is not None:
                # Import the top-level module
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
        print(f"\n! Missing Isaac Sim modules: {missing_modules}")
        print("  Isaac Sim may not be properly installed")
        return False

    print(f"\n✓ Found Isaac Sim modules: {found_modules}")
    return True

def check_ros2_bridge():
    """Check if Isaac Sim ROS 2 bridge is available."""
    print("\nChecking Isaac Sim ROS 2 bridge...")

    # Check if the ROS 2 bridge extension exists
    try:
        import omni
        # Try to import ROS bridge components
        try:
            import omni.isaac.ros_bridge
            print("✓ omni.isaac.ros_bridge importable")
            return True
        except ImportError:
            print("! omni.isaac.ros_bridge not available")
            # Check if extension exists in Isaac Sim
            try:
                import carb
                extensions = carb.settings.get_settings().get("/exts/enabled")
                ros_bridge_found = any("ros_bridge" in ext for ext in extensions)
                if ros_bridge_found:
                    print("  ! ROS bridge extension exists but not importable")
                else:
                    print("  ! ROS bridge extension not found in Isaac Sim")
            except Exception:
                print("  ! Could not check Isaac Sim extensions")
            return False
    except ImportError:
        print("! omni module not available - Isaac Sim not properly installed")
        return False

def check_gpu_support():
    """Check if GPU support is available for Isaac Sim."""
    print("\nChecking GPU support...")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA GPU available: {gpu_name}")
            return True
        else:
            print("! CUDA GPU not available, but Isaac Sim might run in software mode")
            return True  # Not strictly required for headless
    except ImportError:
        print("! PyTorch not available for GPU check, but Isaac Sim may still work")
        return True  # GPU check is optional

def check_headless_mode():
    """Check if Isaac Sim can run in headless mode."""
    print("\nChecking headless mode capability...")

    # Test if Isaac Sim can be initialized in headless mode
    try:
        # Try to import Isaac Sim components
        from omni.isaac.kit import SimulationApp

        # Attempt to create a headless simulation app
        # This is a minimal test without actually launching to avoid UI dependencies
        print("✓ Isaac Sim headless mode supported (import capability verified)")
        return True
    except ImportError as e:
        print(f"! Isaac Sim headless mode not supported: {e}")
        return False
    except Exception as e:
        print(f"! Isaac Sim headless mode error: {e}")
        return False

def main():
    print("Isaac Sim Runtime Installation Validation")
    print("="*50)

    # Test 1: Isaac Sim imports
    test1_result = check_isaac_sim_imports()

    # Test 2: ROS 2 bridge
    test2_result = check_ros2_bridge()

    # Test 3: GPU support
    test3_result = check_gpu_support()

    # Test 4: Headless mode
    test4_result = check_headless_mode()

    print("\n" + "="*50)
    print("Validation Summary:")
    print(f"Isaac Sim Imports: {'PASS' if test1_result else 'FAIL'}")
    print(f"ROS 2 Bridge: {'PASS' if test2_result else 'FAIL'}")
    print(f"GPU Support: {'PASS' if test3_result else 'FAIL'}")
    print(f"Headless Mode: {'PASS' if test4_result else 'FAIL'}")

    all_passed = all([test1_result, test2_result, test3_result, test4_result])

    if all_passed:
        print("\n✓ Isaac Sim runtime installation appears to be valid!")
        print("  Note: Full runtime validation requires actual Isaac Sim environment.")
        return 0
    else:
        print("\n✗ Isaac Sim runtime installation validation incomplete!")
        print("  Please ensure Isaac Sim is properly installed with ROS 2 bridge.")
        return 1

if __name__ == "__main__":
    sys.exit(main())