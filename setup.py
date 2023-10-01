from setuptools import find_packages, setup


def read_file(file):
    with open(file) as f:
        return f.read()


version = read_file("VERSION")

setup(
    name='juspln',
    version=version,
    author="kelsensantos",
    packages=find_packages(exclude=['notebooks']),
    author_email="kelsensantos@gmail.com",
    url='https://github.com/kelsensantos/juspln/',
    description='Modelos pré treinados para aplicação de PLN em textos jurídicos.',
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)
