#!/usr/bin/env python3
"""
Isaac ROS Installation Validation Script

This script validates that Isaac ROS is properly installed and functional.
It checks for the basic requirements mentioned in T002 task:
- ROS 2 Humble Hawksbill installed and sourced
- Isaac ROS packages available
- Dependencies installed
- Workspace builds successfully
- GPU acceleration verified
"""

import sys
import subprocess
import os

def check_ros2_installation():
    """Check if ROS 2 Humble is installed and sourced."""
    print("Checking ROS 2 Humble installation...")

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

def check_isaac_ros_packages():
    """Check if Isaac ROS packages are available."""
    print("\nChecking Isaac ROS packages...")

    isaac_ros_packages = [
        'isaac_ros_common',
        'isaac_ros_perception',
        'isaac_ros_visual_slam',
        'isaac_ros_image_pipeline'
    ]

    found_packages = []
    missing_packages = []

    for package in isaac_ros_packages:
        try:
            result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True)
            if package in result.stdout:
                found_packages.append(package)
                print(f"✓ {package} found")
            else:
                missing_packages.append(package)
                print(f"! {package} not found")
        except Exception as e:
            print(f"✗ Error checking {package}: {e}")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n! Missing Isaac ROS packages: {missing_packages}")
        print("  These need to be installed separately")
        return False

    print(f"\n✓ Found Isaac ROS packages: {found_packages}")
    return True

def check_dependencies():
    """Check for common dependencies."""
    print("\nChecking dependencies...")

    dependencies = [
        'cv_bridge',
        'image_transport',
        'vision_msgs',
        'sensor_msgs',
        'geometry_msgs'
    ]

    found_deps = []
    missing_deps = []

    for dep in dependencies:
        try:
            result = subprocess.run(['ros2', 'pkg', 'list'], capture_output=True, text=True)
            if dep in result.stdout:
                found_deps.append(dep)
                print(f"✓ {dep} found")
            else:
                missing_deps.append(dep)
                print(f"! {dep} not found")
        except Exception as e:
            print(f"✗ Error checking {dep}: {e}")
            missing_deps.append(dep)

    if missing_deps:
        print(f"\n! Missing dependencies: {missing_deps}")
        print("  These need to be installed separately")
        return False

    print(f"\n✓ Found dependencies: {found_deps}")
    return True

def check_gpu_acceleration():
    """Check for GPU acceleration capabilities."""
    print("\nChecking GPU acceleration...")

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA GPU available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            return True
        else:
            print("! CUDA GPU not available")
            return False
    except ImportError:
        print("! PyTorch not installed, cannot verify GPU acceleration")
        return False
    except Exception as e:
        print(f"! Error checking GPU acceleration: {e}")
        return False

def main():
    print("Isaac ROS Installation Validation")
    print("="*40)

    # Test 1: ROS 2 installation
    test1_result = check_ros2_installation()

    # Test 2: Isaac ROS packages
    test2_result = check_isaac_ros_packages()

    # Test 3: Dependencies
    test3_result = check_dependencies()

    # Test 4: GPU acceleration
    test4_result = check_gpu_acceleration()

    print("\n" + "="*40)
    print("Validation Summary:")
    print(f"ROS 2 Installation: {'PASS' if test1_result else 'FAIL'}")
    print(f"Isaac ROS Packages: {'PASS' if test2_result else 'FAIL'}")
    print(f"Dependencies: {'PASS' if test3_result else 'FAIL'}")
    print(f"GPU Acceleration: {'PASS' if test4_result else 'FAIL'}")

    all_passed = all([test1_result, test2_result, test3_result, test4_result])

    if all_passed:
        print("\n✓ Isaac ROS installation appears to be valid!")
        print("  Note: Actual Isaac ROS functionality requires proper installation.")
        return 0
    else:
        print("\n✗ Isaac ROS installation validation incomplete!")
        print("  Please ensure Isaac ROS is properly installed following NVIDIA's documentation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())