from setuptools import find_packages, setup

setup(
    name="kirby",
    version="1.0.0",
    url="https://github.com/rmorain/kirby",
    author="Robert Morain",
    author_email="robert.morain@gmail.com",
    description="GPT-2 and Wikidata",
    packages=find_packages(),
    install_requires=["torch", "transformers"],
)
