import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="schedcat",
    version="0.0.1",
    author="Bjorn Brandenburg",
    author_email="bbb@mpi-sws.org",
    description="schedcat branch for GIPP",
    long_description="schedcat branch for GIPP",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=2.7, <3  ',
)
