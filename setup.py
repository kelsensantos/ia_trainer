import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="iatrainer",
    version="1",
    author="Kelsen Henrique Roim dos Santos",
    author_email="kelsensantos@gmail.com",
    description="Modelos pré treinados para aplicação de PLN em textos jurídicos.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/kelsensantos/ia_trainer',
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    # install_requires=['ftfy','wget']
)
