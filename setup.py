from setuptools import find_packages, setup

package_name = 'track'

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
    maintainer='ros_dev',
    maintainer_email='ros_dev@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'bytetrack = track.bytetrack:main',
            'byte_old = track.byte_old:main',
            'test = track.test:main',
            'track_point_sub = track.track_point_sub:main',
            # 'trans_depth = track.depth:main'
            'open_rs = track.open_rs:main'
        ],
    },
)
