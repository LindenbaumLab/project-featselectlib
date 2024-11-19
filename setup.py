from setuptools import setup, find_packages
from pathlib import Path

# read the contents of your existing README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding='utf-8')  # Change to README.rst if that's your file

setup(
    name='featselectlib',
    version='0.1.0',
    description='A library combining different feature selection methods',
    long_description=long_description,
    long_description_content_type='text/markdown',  # Change to 'text/x-rst' if using .rst
    author='LindenbaumLab',
    author_email='your.email@example.com',
    url='https://github.com/LindenbaumLab/project-featselectlib',
    packages=find_packages(),
    install_requires=[
        'torch',
        'scikit-learn',
        'omegaconf',
        'scipy',
        'matplotlib'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)

