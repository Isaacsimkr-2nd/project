from setuptools import find_packages, setup

package_name = 'turtle_llm'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
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
            'llm_node = turtle_llm.llm_node:main',
            'response_node = turtle_llm.response_node:main',
        ],
    },
)
