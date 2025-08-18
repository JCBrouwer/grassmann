from setuptools import setup

setup(
    name="grassmann",  # project name
    version="0.2",
    description="Implementation of binary distributions in the Grassmann formalism, including conditional distributions and estimating methods.",
    url="https://github.com/JCBrouwer/grassmann.git",
    author="Hans Brouwer",
    author_email="hans@brouwer.work",
    license="MIT",
    packages=["grassmann"],  # actual package name (to import package)
    zip_safe=False,
)
