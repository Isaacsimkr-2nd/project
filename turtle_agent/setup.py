from setuptools import find_packages, setup
from glob import glob
import os


package_name = 'turtle_agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=['turtle_agent', 'turtle_agent.Agents'], 
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='khw',
    maintainer_email='khw11044@pinklab.art',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'main_node = turtle_agent.main_node:main',
            'response_node = turtle_agent.response_node:main',
        ],
    },
)

