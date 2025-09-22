"""Setup configuration for Toaripi SLM package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "readme.md"
if readme_path.exists():
    with open(readme_path, "r", encoding="utf-8") as fh:
        long_description = fh.read()
else:
    long_description = "A small language model for generating educational content in Toaripi language"

# Read requirements
def read_requirements(filename):
    """Read requirements from file."""
    req_path = Path(__file__).parent / filename
    if req_path.exists():
        with open(req_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip() and not line.startswith("#")]
    return []

setup(
    name="toaripi-slm",
    version="0.1.0",
    author="Toaripi SLM Contributors",
    author_email="ck@raposo.ai",
    description="A small language model for generating educational content in Toaripi language",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/cknzraposo/toaripi-slm",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=read_requirements("requirements.txt"),
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0", 
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "jupyter>=1.0.0"
        ],
        "gpu": ["torch>=2.0.0"],
        "gguf": ["llama-cpp-python>=0.1.78"],
    },
    entry_points={
        "console_scripts": [
            "toaripi-slm=toaripi_slm.cli.main:cli",
            "toaripi-prepare-data=scripts.prepare_data:main",
            "toaripi-finetune=scripts.finetune:main", 
            "toaripi-generate=scripts.generate:main",
            "toaripi-export=scripts.export_gguf:main",
            "toaripi-serve=app.server:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Education", 
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
        "Topic :: Text Processing :: Linguistic",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Natural Language :: English",
        "Natural Language :: Other",
    ],
    keywords=[
        "language-model",
        "toaripi",
        "education",
        "nlp",
        "papua-new-guinea",
        "endangered-languages",
        "language-preservation",
        "machine-learning",
        "transformers",
    ],
    project_urls={
        "Bug Reports": "https://github.com/cknzraposo/toaripi-slm/issues",
        "Source": "https://github.com/cknzraposo/toaripi-slm",
        "Documentation": "https://github.com/cknzraposo/toaripi-slm/docs",
    },
)