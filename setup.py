from setuptools import find_packages, setup

setup(
    name='mlp-python',
    packages=find_packages(include=['mlp-python']),
    version='0.0.1',
    description='A Python library to create simple Multy Layer Perceptron based neural networks.',
    author='Philipp Bleimund',
    install_requires=['numpy'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest==4.4.1'],
    test_suite='tests',
    extras_require={
        "dev": [
            "pytest",
            "keras",
            "tensorflow"
        ],
    },
)
