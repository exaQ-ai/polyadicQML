import setuptools

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup_kwargs = dict(
    name="polyadicqml",
    version="0.1.0b3",
    author="William Cappelletti",
    author_email="cappelletti.william@gmail.com",
    description="High level API to define, train and deploy Polyadic Quantum Machine Learning models",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/entropicalabs/polyadicQML",
    packages=setuptools.find_packages(
        include=('polyadicqml', 'polyadicqml.*')
    ),
    license='Apache 2.0',
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
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

if __name__ == '__main__':
    setuptools.setup(**setup_kwargs)
