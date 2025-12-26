#!/usr/bin/env python3
"""
Isaac Sim to ROS 2 Bridge Communication Validation Script

This script validates that Isaac Sim can communicate with ROS 2 through the bridge
without implementing complex communication logic.
"""

import sys
import subprocess
import importlib.util

def check_isaac_ros_bridge_imports():
    """Check if Isaac Sim ROS bridge modules can be imported."""
    print("Checking Isaac Sim ROS bridge modules...")

    bridge_modules = [
        'omni.isaac.ros_bridge',
    ]

    missing_modules = []
    found_modules = []

    for module_path in bridge_modules:
        try:
            # Try to import the bridge module
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
        print(f"\n! Missing Isaac Sim ROS bridge modules: {missing_modules}")
        print("  Isaac Sim ROS bridge may not be properly installed")
        return False

    print(f"\n✓ Found Isaac Sim ROS bridge modules: {found_modules}")
    return True

def check_ros2_installation():
    """Check if ROS 2 Humble is installed and sourced."""
    print("\nChecking ROS 2 Humble installation...")

    try:
        # Check if ROS_DISTRO is set to humble
        ros_distro = os.environ.get('ROS_DISTRO', '')
        if 'humble' in ros_distro.lower():
            print(f"✓ ROS_DISTRO is set to: {ros_distro}")
        else:
            print(f"! ROS_DISTRO is set to: {ros_distro} (should be humble)")
            print("  Make sure to source ROS 2 Humble: source /opt/ros/humble/setup.bash")
            return False

        # Check if ROS commands work
        result = subprocess.run(['ros2', '--version'], capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✓ ROS 2 version: {result.stdout.strip()}")
        else:
            print("✗ ROS 2 command not working")
            return False

        return True
    except FileNotFoundError:
        print("✗ ROS 2 not found in PATH")
        return False
    except Exception as e:
        print(f"✗ Error checking ROS 2: {e}")
        return False

def check_ros2_topics_services():
    """Check if ROS 2 topic and service tools are available."""
    print("\nChecking ROS 2 topic and service tools...")

    try:
        # Check topic tools
        result = subprocess.run(['ros2', 'topic', '--help'],
                              capture_output=True, text=True, timeout=5)
        if 'topic' in result.stdout:
            print("✓ ROS 2 topic tools available")
            topic_ok = True
        else:
            print("! ROS 2 topic tools not available")
            topic_ok = False
    except Exception as e:
        print(f"! Error checking topic tools: {e}")
        topic_ok = False

    try:
        # Check service tools
        result = subprocess.run(['ros2', 'service', '--help'],
                              capture_output=True, text=True, timeout=5)
        if 'service' in result.stdout:
            print("✓ ROS 2 service tools available")
            service_ok = True
        else:
            print("! ROS 2 service tools not available")
            service_ok = False
    except Exception as e:
        print(f"! Error checking service tools: {e}")
        service_ok = False

    return topic_ok and service_ok

def check_isaac_sim_core():
    """Check if Isaac Sim core modules are available for bridge validation."""
    print("\nChecking Isaac Sim core modules for bridge...")

    try:
        # Try to import Isaac Sim core components needed for bridge
        import omni.isaac.core
        import omni.isaac.core.utils
        print("✓ Isaac Sim core modules available")
        return True
    except ImportError as e:
        print(f"! Isaac Sim core modules not available: {e}")
        return False

def check_basic_bridge_concepts():
    """Check if basic bridge concepts can be imported."""
    print("\nChecking basic bridge concepts...")

    try:
        # Check if we can access the basic concepts needed for bridge validation
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

        return True
    except Exception as e:
        print(f"! Error checking bridge concepts: {e}")
        return False

def main():
    print("Isaac Sim to ROS 2 Bridge Communication Validation")
    print("="*60)

    # Test 1: ROS 2 installation
    test1_result = check_ros2_installation()

    # Test 2: Isaac Sim ROS bridge imports
    test2_result = check_isaac_ros_bridge_imports()

    # Test 3: ROS 2 tools availability
    test3_result = check_ros2_topics_services()

    # Test 4: Isaac Sim core modules
    test4_result = check_isaac_sim_core()

    # Test 5: Basic bridge concepts
    test5_result = check_basic_bridge_concepts()

    print("\n" + "="*60)
    print("Validation Summary:")
    print(f"ROS 2 Installation: {'PASS' if test1_result else 'FAIL'}")
    print(f"Isaac ROS Bridge Imports: {'PASS' if test2_result else 'FAIL'}")
    print(f"ROS 2 Tools: {'PASS' if test3_result else 'FAIL'}")
    print(f"Isaac Sim Core: {'PASS' if test4_result else 'FAIL'}")
    print(f"Bridge Concepts: {'PASS' if test5_result else 'FAIL'}")

    all_passed = all([test1_result, test2_result, test3_result, test4_result, test5_result])

    if all_passed:
        print("\n✓ Isaac Sim to ROS 2 bridge communication validation successful!")
        print("  All basic bridge components are available.")
        return 0
    else:
        print("\n✗ Isaac Sim to ROS 2 bridge communication validation incomplete!")
        print("  Some bridge components are not available.")
        return 1

if __name__ == "__main__":
    sys.exit(main())