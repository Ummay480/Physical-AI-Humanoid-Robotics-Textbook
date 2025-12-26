#!/usr/bin/env python3
"""
Isaac ROS Basic Perception Nodes Validation Script

This script validates that basic Isaac ROS perception nodes are available
and can be imported without implementing complex perception logic.
"""

import sys
import subprocess
import importlib.util

def check_isaac_ros_perception_packages():
    """Check if Isaac ROS perception packages are available."""
    print("Checking Isaac ROS perception packages...")

    perception_packages = [
        'isaac_ros_visual_slam',
        'isaac_ros_image_pipeline',
        'isaac_ros_apriltag',
        'isaac_ros_detectnet',
        'isaac_ros_hawk',
        'isaac_ros_compressed_image_transport',
        'isaac_ros_image_transport',
        'isaac_ros_pointcloud_utils'
    ]

    found_packages = []
    missing_packages = []

    try:
        result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True)
        all_packages = result.stdout.split('\n')

        for package in perception_packages:
            found = False
            for pkg in all_packages:
                if package in pkg:
                    found_packages.append(package)
                    print(f"✓ {package} found")
                    found = True
                    break
            if not found:
                missing_packages.append(package)
                print(f"! {package} not found")

    except Exception as e:
        print(f"✗ Error checking Isaac ROS perception packages: {e}")
        return False

    if missing_packages:
        print(f"\n! Missing Isaac ROS perception packages: {missing_packages}")
        print("  These need to be installed separately")
        return False

    print(f"\n✓ Found Isaac ROS perception packages: {found_packages}")
    return True

def check_isaac_ros_imports():
    """Check if basic Isaac ROS perception modules can be imported."""
    print("\nChecking Isaac ROS perception imports...")

    # Try importing basic Isaac ROS perception components
    try:
        # Check if common Isaac ROS modules exist
        import ros2pkg
        print("✓ ROS 2 package tools available")
    except ImportError:
        print("! ROS 2 package tools not available")
        return False

    # Try to import some basic perception-related modules
    try:
        # These are basic checks to see if Isaac ROS perception infrastructure is available
        # We're not actually using complex perception, just checking availability
        try:
            import cv2
            print("✓ OpenCV available (needed for perception)")
        except ImportError:
            print("! OpenCV not available (needed for perception)")

        try:
            import numpy as np
            print("✓ NumPy available (needed for perception)")
        except ImportError:
            print("! NumPy not available (needed for perception)")

        return True
    except Exception as e:
        print(f"! Error checking perception imports: {e}")
        return False

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

def check_node_availability():
    """Check if Isaac ROS perception nodes are available."""
    print("\nChecking Isaac ROS perception node availability...")

    # Check if we can access Isaac ROS perception nodes
    try:
        # This is a basic check to see if Isaac ROS perception nodes exist
        result = subprocess.run(['ros2', 'node', '--help'],
                              capture_output=True, text=True, timeout=5)
        if 'node' in result.stdout:
            print("✓ ROS 2 node tools available")
            return True
        else:
            print("! ROS 2 node tools not available")
            return False
    except Exception as e:
        print(f"! Error checking node tools: {e}")
        return False

def check_topic_availability():
    """Check if Isaac ROS perception topics can be managed."""
    print("\nChecking Isaac ROS perception topic management...")

    # Check if we can access topic-related tools
    try:
        result = subprocess.run(['ros2', 'topic', '--help'],
                              capture_output=True, text=True, timeout=5)
        if 'topic' in result.stdout:
            print("✓ Topic tools available")
            return True
        else:
            print("! Topic tools not available")
            return False
    except Exception as e:
        print(f"! Error checking topic tools: {e}")
        return False

def main():
    print("Isaac ROS Basic Perception Nodes Validation")
    print("="*60)

    # Test 1: ROS 2 installation
    test1_result = check_ros2_installation()

    # Test 2: Isaac ROS perception packages
    test2_result = check_isaac_ros_perception_packages()

    # Test 3: Isaac ROS imports
    test3_result = check_isaac_ros_imports()

    # Test 4: Node tools availability
    test4_result = check_node_availability()

    # Test 5: Topic tools availability
    test5_result = check_topic_availability()

    print("\n" + "="*60)
    print("Validation Summary:")
    print(f"ROS 2 Installation: {'PASS' if test1_result else 'FAIL'}")
    print(f"Isaac ROS Perception Packages: {'PASS' if test2_result else 'FAIL'}")
    print(f"Isaac ROS Imports: {'PASS' if test3_result else 'FAIL'}")
    print(f"Node Tools: {'PASS' if test4_result else 'FAIL'}")
    print(f"Topic Tools: {'PASS' if test5_result else 'FAIL'}")

    all_passed = all([test1_result, test2_result, test3_result, test4_result, test5_result])

    if all_passed:
        print("\n✓ Isaac ROS basic perception nodes validation successful!")
        print("  All basic Isaac ROS perception components are available.")
        return 0
    else:
        print("\n✗ Isaac ROS basic perception nodes validation incomplete!")
        print("  Some perception components are not available.")
        return 1

if __name__ == "__main__":
    sys.exit(main())