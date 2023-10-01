from setuptools import find_packages, setup


def read_file(file):
    with open(file) as f:
        return f.read()


def read_requirements(file):
    with open(file) as f:
        return f.read().splitlines()


version = read_file("VERSION")
requirements = read_requirements("requirements.txt")


setup(
    name='juspln',
    version=version,
    author="kelsensantos",
    packages=['jusnlp'],
    author_email="kelsensantos@gmail.com",
    url='https://github.com/kelsensantos/juspln/',
    description='Modelos pré treinados para aplicação de PLN em textos jurídicos.',
    license="MIT",
    install_requires=requirements,
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
