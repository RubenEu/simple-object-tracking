from setuptools import setup, find_packages

setup(
    name='simple-object-tracking',
    version='0.1.1',
    description='Object tracker with different models and parameters.',
    author='Rubén García Rojas',
    author_email='garcia.ruben@outlook.es',
    packages=find_packages(),
    install_requires=[
        'simple-object-detection',
        'opencv-python',
        'numpy',
    ]
)
