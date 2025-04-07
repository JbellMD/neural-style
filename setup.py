"""
Setup script for neural style transfer application.
"""

from setuptools import setup, find_packages

setup(
    name="neural-style",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=1.9.0",
        "torchvision>=0.10.0",
        "numpy>=1.19.5",
        "Pillow>=8.2.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.2",
        "tqdm>=4.61.1",
        "Flask>=2.0.1",
        "Flask-Cors>=3.0.10",
        "Werkzeug>=2.0.1",
        "requests>=2.25.1",
        "opencv-python>=4.5.2",
        "pyyaml>=5.4.1",
        "click>=8.0.1",
        "loguru>=0.5.3",
    ],
    entry_points={
        "console_scripts": [
            "neural-style=neural_style.__main__:main",
        ],
    },
    python_requires=">=3.7",
    author="Your Name",
    author_email="your.email@example.com",
    description="Neural Style Transfer Application",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/neural-style",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
