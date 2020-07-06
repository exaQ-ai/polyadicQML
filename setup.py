import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polyadicqml",
    version="0.1.0.dev0",
    author="William Cappelletti",
    author_email="cappelletti.william@gmail.com",
    description="Poliadic Quantum Machine Learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/entropica/polyadic-quantum-ml",
    packages=setuptools.find_packages(
        exclude=('examples')
    ),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 1 - Planning",
    ],
    install_requires=[
        'numpy',
        'scipy',
        'sklearn',
        'qiskit',
        'manyq',
        'tqdm'
    ],
    python_requires='>=3.6',
)