import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="fusionrate",
    version="0.0.1",
    author="Jacob Schwartz",
    author_email="jacob@jaschwartz.net",
    description="Fusion cross sections and rate coefficients",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PlasmaControl/fusionrate",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=[
        "cubature >= 0.14.5",
        "numpy >= 1.22.4",
        "scipy >= 1.8.1",
        "h5py >= 3.7.0",
    ],
    include_package_data=True,
    package_data = {
        '': ['*.csv'],
    },
)
