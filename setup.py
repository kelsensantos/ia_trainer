from setuptools import find_packages, setup

# with open("app/README.md", "r") as f:
#     long_description = f.read()

setup(
    name="juspln",
    version="1.0.0",
    description="Modelos pré treinados para aplicação de PLN em textos jurídicos.",
    package_dir={"": "juspln"},
    packages=find_packages(where="juspln"),
    # long_description=long_description,
    # long_description_content_type="text/markdown",
    url="https://github.com/kelsensantos/juspln",
    author="kelsensantos",
    author_email="kelsensantos@gmail.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    install_requires=[],
    extras_require={
        # "dev": ["pytest>=7.0", "twine>=4.0.2"],
    },
    python_requires=">=3.9",
)

