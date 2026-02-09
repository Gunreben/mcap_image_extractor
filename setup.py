from setuptools import find_packages, setup

package_name = 'mcap_image_extractor'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=[
        'setuptools',
        'mcap',
        'mcap-ros2-support',
        'numpy',
        'opencv-python',
        'PySide6',
        'qt-material',
    ],
    zip_safe=True,
    maintainer='User',
    maintainer_email='user@example.com',
    description='Extract images from ROS2 MCAP bags for CVAT labeling',
    license='MIT',
    entry_points={
        'console_scripts': [
            'mcap_image_extractor = mcap_image_extractor.main:main',
            'mcap_image_extractor_gui = mcap_image_extractor.gui:main',
        ],
    },
)
