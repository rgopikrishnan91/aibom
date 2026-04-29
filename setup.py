"""
BOM Tools - AI Model and Dataset BOM Generator
"""
from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="aikaboom",
    version="1.0.0",
    author="AIkaBoOM Contributors",
    author_email="",
    description="Unified AI Model and Dataset BOM Generator with RAG",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rgopikrishnan91/aibom",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=[
        "flask>=3.0.0",
        "openai>=1.0.0",
        "python-dotenv>=1.0.0",
        "langchain>=0.1.0",
        "langchain-openai>=0.0.5",
        "langchain-community>=0.0.20",
        "langchain-huggingface>=0.0.1",
        "langchain-text-splitters>=0.0.1",
        "langchain-core>=0.1.0",
        "langgraph>=0.0.20",
        "faiss-cpu>=1.7.4",
        "sentence-transformers>=2.2.0",
        "chromadb>=0.4.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "PyGithub>=2.1.1",
        "huggingface-hub>=0.20.0",
        "requests>=2.31.0",
        "beautifulsoup4>=4.12.0",
        "pymupdf>=1.23.0",
        "certifi>=2023.0.0",
        "google-genai>=0.3.0",
        "httpx>=0.25.0",
    ],
    extras_require={
        "link_fallback": [
            "google-genai>=0.3.0",
            "httpx>=0.25.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aikaboom=aikaboom.cli:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aikaboom.web": ["templates/*.html", "static/*"],
    },
)
