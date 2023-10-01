from setuptools import find_packages, setup


with open("README.md", "r") as f:
    long_description = f.read()


def read_file(file):
    with open(file) as f:
        return f.read()


version = read_file("VERSION")

setup(
    name='juspln',
    version=version,
    description='Modelos pré treinados para aplicação de PLN em textos jurídicos.',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kelsensantos/juspln/',
    author="kelsensantos",
    author_email="kelsensantos@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
)
