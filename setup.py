from setuptools import setup, find_packages

setup(
    name="generative-ai-training",
    version="0.1.0",
    description="A comprehensive framework for training generative AI models",
    author="Rachel Ouk",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.21.0",
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "pillow>=8.3.0",
        "tqdm>=4.62.0",
        "tensorboard>=2.8.0",
        "pyyaml>=6.0",
        "scikit-learn>=1.0.0",
        "jupyter>=1.0.0",
        "notebook>=6.0.0",
    ],
    python_requires=">=3.8",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)