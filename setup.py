from setuptools import setup, find_packages

setup(
    name="gym_woodoku_fast",
    version="0.1",
    description="Efficient python implementation of Woodoku environment.",
    author="Mark Knyazev",
    author_email="mark.knyazev4@gmail.com",
    packages=find_packages(),
    install_requires=[
        "cloudpickle==3.0.0",
        "Farama-Notifications==0.0.4",
        "gymnasium==0.29.1",
        "numpy==2.1.1",
        "typing_extensions==4.12.2"
    ],
)