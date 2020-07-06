import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="polyadicqml",
    version="0.1.0.dev0",
    author="William Cappelletti",
    author_email="cappelletti.william@gmail.com",
    description="Poliadic Quantum Machine Learning",
    long_description=long_description,
    long_description_content_type="text/x-rst",
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
        'manyq @ git+https://gitlab.com/entropica/manyq#egg=manyq'
        'tqdm'
    ],
    # dependency_links=['http://github.com/user/repo/tarball/master#egg=package-1.0'],
    python_requires='>=3.6',
)