from setuptools import setup

setup(
    name="improved-diffusion",
    version='7.0',
    py_modules=["improved_diffusion"],
    install_requires=["blobfile>=1.0.5", "torch", "tqdm"],
)
