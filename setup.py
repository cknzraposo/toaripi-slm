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


def read_requirements(filename: str):
    """Read requirements from a file, ignoring comments, blanks, and recursive includes.

    Recursive includes (e.g. '-r requirements.txt') are intentionally skipped because:
    - Base dependencies are already declared in install_requires.
    - extras_require values must be concrete requirement specifiers.
    """
    req_path = Path(__file__).parent / filename
    if not req_path.exists():
        return []
    requirements = []
    with open(req_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith(("-r", "--requirement")):
                # Skip recursive includes for packaging metadata
                continue
            requirements.append(line)
    return requirements


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
        # Development / QA / docs / notebooks
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "pytest-asyncio>=0.21.0",
            "pytest-benchmark>=4.0.0",
            "black>=23.7.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
            "bandit>=1.7.5",
            "pre-commit>=3.3.0",
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
            "jupyter>=1.0.0",
            "notebook>=6.5.0",
            "ipykernel>=6.25.0",
        ],
        # Smaller docs-only set if someone prefers: pip install .[docs]
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
        ],
        # llama.cpp Python bindings for GGUF export / CPU quant usage
        "gguf": [
            "llama-cpp-python>=0.1.78",
        ],
        # GPU convenience (users can still install specific CUDA wheels manually)
        "gpu": [
            "torch>=2.0.0",
        ],
    },
    # entry_points={
    #     "console_scripts": [
    #         "toaripi-prepare-data=scripts.prepare_data:main",
    #         "toaripi-finetune=scripts.finetune:main",
    #         "toaripi-generate=scripts.generate:main",
    #         "toaripi-export=scripts.export_gguf:main",
    #         "toaripi-serve=app.server:main",
    #     ],
    # },
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
    license="MIT",
)