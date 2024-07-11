from setuptools import setup, find_packages

setup(
    name='featselectlib',
    version='0.1',
    description='A library combining STG and LSCAE for feature selection',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/featselectlib',  # Change to your repository URL
    packages=find_packages(),
    install_requires=[
                    'torch',
                    'scikit-learn',
                    'omegaconf',
                    'scipy',
                    'matplotlib'
        # Add other dependencies if needed
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
