import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="sched_experiments",
    version="0.0.1",
    author="Bjorn Brandeburg",
    author_email="bbb@mpi-sws.org", 
    description="Automation of experiments for SchedCAT",
    long_description="Automation of experiments for SchedCAT",
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    python_requires='>=2.7, <3  ',
)