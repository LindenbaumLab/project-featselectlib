from setuptools import setup, find_packages

setup(
    name='featselectlib',
    version='0.1',
    description='A library combining different feature selection methods',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/featselectlib', 
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
