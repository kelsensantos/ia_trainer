import setuptools

setuptools.setup(
    name="iatrainer",
    version="1.0.0",
    author="Kelsen Henrique Rolim dos Santos",
    author_email="kelsensantos@gmail.com",
    description="Modelos pré treinados para aplicação de PLN em textos jurídicos.",
    url='https://github.com/kelsensantos/ia_trainer',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)

from setuptools import find_packages, setup

with open("app/README.md", "r") as f:
    long_description = f.read()

setup(
    name="idgenerator",
    version="0.0.10",
    description="An id generator that generated various types and lengths ids",
    package_dir={"": "app"},
    packages=find_packages(where="app"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ArjanCodes/2023-package",
    author="ArjanCodes",
    author_email="arjan@arjancodes.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=["bson >= 0.5.10"],
    extras_require={
        "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.10",
)