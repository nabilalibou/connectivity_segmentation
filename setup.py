from setuptools import find_packages, setup

setup(
    name="connectivity-segmentation",
    version="0.0.1",
    url="https://github.com/Nabil-AL/connectivity_segmentation.git",
    packages=find_packages(),
    license="MIT License",
    author="Nabil ALIBOU",
    description="Library to track the spatiotemporal dynamics of brain network based on a modified k-means"
    "clustering algorithm adapted to EEG connectivity graphs.",
)
