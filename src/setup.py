import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="boiler-plate-pytorch", 
    version="0.0.5",
    author="Bipin Krishnan P",
    author_email="bipinkrishna.p@gmail.com",
    description="A python package that contains PyTorch boilerplate code for different use cases",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bipinKrishnan/boiler_plate",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
