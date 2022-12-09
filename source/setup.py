from setuptools import find_packages, setup

setup(
    name='text_analysis',
    packages=find_packages(include=['text_analysis']),
    version='0.1.0',
    description='A little library for text analysis.',
    author='Flavio Baccarini',
    license='MIT',
    install_requires=['nltk',
                      'pandas', 
                      'numpy', 
                      'gensim',
                      'tensorflow',
                      'scikit-learn',
                      'seaborn',
                      'matplotlib',
                      'pathlib',
                      'hypothesis',
                      'pytest']
)
