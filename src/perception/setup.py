from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='maintainer',
    maintainer_email='maintainer@todo.todo',
    description='Perception system for humanoid robots with sensor acquisition, computer vision, and sensor fusion capabilities',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
        ],
    },
)