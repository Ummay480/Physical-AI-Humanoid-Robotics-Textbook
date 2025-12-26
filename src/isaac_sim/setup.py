from setuptools import setup
import os

package_name = 'isaac_sim'

setup(
    name=package_name,
    version='1.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='AI-Robot Brain Team',
    maintainer_email='ai-robot-brain@example.com',
    description='NVIDIA Isaac Sim package for photorealistic simulation and synthetic data generation',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)