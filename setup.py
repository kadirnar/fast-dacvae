from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()

setup(
    name="fast-dacvae",
    version="1.0.0",
    description="Optimized DACVAE inference — 3.6x faster on H100 GPU.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Kadir Nar",
    url="https://github.com/kadirnar/fast-dacvae",
    license="Apache-2.0",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "einops",
        "numpy",
        "torch>=2.4",
        "huggingface-hub",
    ],
    extras_require={
        "tensorrt": [
            "torch-tensorrt",
        ],
    },
)
