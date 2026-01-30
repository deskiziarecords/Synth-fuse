#!/usr/bin/env python3
"""
Synth-Fuse v0.2.0 - Unified Field Engineering
Production-ready packaging setup.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Any

from setuptools import setup, find_packages
from setuptools.command.build_ext import build_ext
from setuptools.extension import Extension
import subprocess

# --------------------------------------------------------------
# Configuration
# --------------------------------------------------------------

PACKAGE_NAME = "synthfuse"
VERSION = "0.2.0"
AUTHOR = "J. Roberto Jiménez"
AUTHOR_EMAIL = "tijuanapaint@gmail.com"
DESCRIPTION = "Unified Field Engineering – A Deterministic Hybrid Organism Architecture"
LONG_DESCRIPTION = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")
URL = "https://github.com/tijuanapaint/synth-fuse"
LICENSE = "OpenGate Integrity License"

# Python requirements
PYTHON_REQUIRES = ">=3.10"

# --------------------------------------------------------------
# Rust Extension Configuration
# --------------------------------------------------------------

class RustExtension(Extension):
    """Custom extension for Rust library."""
    
    def __init__(self, name, path, features=None):
        self.path = path
        self.features = features or []
        super().__init__(name, sources=[])

class RustBuild(build_ext):
    """Build Rust extension using maturin."""
    
    def run(self):
        # Check for maturin
        try:
            subprocess.run(["maturin", "--version"], 
                         capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("Installing maturin...")
            subprocess.run([sys.executable, "-m", "pip", "install", "maturin"], 
                         check=True)
        
        # Build Rust extension for each platform
        for ext in self.extensions:
            if isinstance(ext, RustExtension):
                self.build_rust_extension(ext)
    
    def build_rust_extension(self, ext: RustExtension):
        """Build a single Rust extension."""
        print(f"Building Rust extension: {ext.name}")
        
        # Determine build options
        cmd = ["maturin", "develop", "--manifest-path", ext.path]
        
        # Add features if specified
        if ext.features:
            cmd.extend(["--features", ",".join(ext.features)])
        
        # Build in release mode for production
        if not self.debug:
            cmd.append("--release")
        
        # Execute build
        try:
            subprocess.run(cmd, check=True, cwd=Path(ext.path).parent)
            print(f"✅ Successfully built {ext.name}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to build {ext.name}: {e}")
            sys.exit(1)

# --------------------------------------------------------------
# Package Discovery
# --------------------------------------------------------------

def find_package_data() -> Dict[str, List[str]]:
    """Find non-Python files to include in package."""
    package_data = {
        "synthfuse": [
            "py.typed",  # For type checking
        ],
        "synthfuse.alchemj": [
            "grammar.lark",  # Lark grammar file
        ],
        "synthfuse.docs": [
            "*.pdf",  # Documentation PDFs
        ],
    }
    
    # Add security certificates
    security_dir = Path("src/synthfuse/security")
    if security_dir.exists():
        cert_files = list(security_dir.glob("*.crt")) + list(security_dir.glob("*.pem"))
        if cert_files:
            package_data["synthfuse.security"] = ["*.crt", "*.pem"]
    
    return package_data

def find_entry_points() -> Dict[str, List[str]]:
    """Find console scripts and entry points."""
    return {
        "console_scripts": [
            "synthfuse = synthfuse.__main__:main",
            "sfbench = synthfuse.tools.bench:main",
            "sfwatch = synthfuse.ingest.watcher:main",
            "sfcabinet = synthfuse.cabinet.cabinet_orchestrator:cli_main",
        ],
        "gui_scripts": [
            "synthfuse-lab = synthfuse.lab.app:main",
        ],
    }

def find_extensions() -> List[Extension]:
    """Find C/C++/Rust extensions."""
    extensions = []
    
    # Rust security core
    rust_manifest = Path("src/rust/Cargo.toml")
    if rust_manifest.exists():
        extensions.append(
            RustExtension(
                name="synthfuse._security_core",
                path=str(rust_manifest),
                features=["python"],  # Enable Python bindings
            )
        )
    
    # Optional: Add Cython extensions in the future
    # extensions.append(
    #     Extension(
    #         "synthfuse._vector_ops",
    #         sources=["src/synthfuse/vector/_vector_ops.pyx"],
    #         libraries=["m"],  # math library
    #     )
    # )
    
    return extensions

# --------------------------------------------------------------
# Dependency Management
# --------------------------------------------------------------

def parse_requirements() -> Dict[str, List[str]]:
    """Parse requirements from various files."""
    requirements = {
        "core": [],
        "dev": [],
        "lab": [],
        "notebook": [],
        "security": [],
    }
    
    # Core runtime dependencies
    requirements["core"] = [
        # JAX ecosystem
        "jax>=0.4.30",
        "jaxlib>=0.4.30",
        "chex>=0.1.8",
        
        # Numerical computing
        "numpy>=1.24.0",
        "scipy>=1.10.0",
        
        # Async and I/O
        "aiofiles>=23.0.0",
        "aiohttp>=3.9.0",
        "watchfiles>=0.20.0",
        
        # Data serialization
        "msgpack>=1.0.5",
        "pyyaml>=6.0",
        "orjson>=3.9.0",
        
        # Configuration
        "pydantic>=2.5.0",
        "pydantic-settings>=2.1.0",
        
        # Logging and monitoring
        "structlog>=23.0.0",
        "prometheus-client>=0.19.0",
        
        # Cryptography (OpenGate)
        "cryptography>=42.0.0",
        "pyOpenSSL>=23.2.0",
        
        # Process management
        "psutil>=5.9.0",
    ]
    
    # Development dependencies
    requirements["dev"] = [
        # Testing
        "pytest>=7.4.0",
        "pytest-cov>=4.1.0",
        "pytest-asyncio>=0.21.0",
        "pytest-xdist>=3.5.0",
        
        # Code quality
        "black>=23.11.0",
        "mypy>=1.7.0",
        "ruff>=0.1.0",
        "pre-commit>=3.5.0",
        
        # Documentation
        "sphinx>=7.2.0",
        "sphinx-rtd-theme>=1.3.0",
        "myst-parser>=2.0.0",
        
        # Build and packaging
        "build>=1.0.0",
        "twine>=4.0.0",
        "wheel>=0.42.0",
        "setuptools>=68.0.0",
        "setuptools-rust>=1.8.0",
        "maturin>=1.4.0",
        
        # Type stubs
        "types-PyYAML>=6.0.0",
        "types-requests>=2.31.0",
    ]
    
    # Lab dependencies (interactive features)
    requirements["lab"] = [
        "fastapi>=0.104.0",
        "uvicorn[standard]>=0.24.0",
        "websockets>=12.0",
        "jinja2>=3.1.0",
        "starlette>=0.34.0",
        
        # Web interface
        "streamlit>=1.28.0",
        "plotly>=5.18.0",
        "dash>=2.14.0",
        
        # Thebe integration
        "jupyter-server>=2.7.0",
        "jupyter-client>=8.6.0",
    ]
    
    # Notebook dependencies
    requirements["notebook"] = [
        "jupyterlab>=4.0.0",
        "ipywidgets>=8.1.0",
        "ipykernel>=6.25.0",
        "matplotlib>=3.8.0",
        "seaborn>=0.13.0",
        "pandas>=2.1.0",
    ]
    
    # Security dependencies (OpenGate)
    requirements["security"] = [
        "cryptography>=42.0.0",
        "pyca/cryptography",  # For OpenGate
        "python-jose[cryptography]>=3.3.0",
        "passlib[bcrypt]>=1.7.4",
    ]
    
    return requirements

# --------------------------------------------------------------
# Setup Configuration
# --------------------------------------------------------------

setup(
    # Basic metadata
    name=PACKAGE_NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    
    # Classifiers for PyPI
    classifiers=[
        # Development Status
        "Development Status :: 4 - Beta",
        
        # Intended Audience
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        
        # Topics
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Software Development :: Libraries",
        
        # License
        "License :: Other/Proprietary License",
        
        # Programming Languages
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Rust",
        
        # Operating Systems
        "Operating System :: OS Independent",
        "Operating System :: POSIX",
        "Operating System :: MacOS :: MacOS X",
        "Operating System :: Microsoft :: Windows",
        
        # Frameworks
        "Framework :: Jupyter",
        "Framework :: FastAPI",
        
        # Typing
        "Typing :: Typed",
    ],
    
    # Keywords for discovery
    keywords=[
        "neurosymbolic",
        "jax",
        "hybrid-intelligence",
        "fusion-calculus",
        "alchemj",
        "stcl",
        "ntep",
        "opengate",
        "deterministic-ai",
        "thermodynamic-computing",
    ],
    
    # Package structure
    packages=find_packages(
        where="src",
        include=["synthfuse", "synthfuse.*"],
        exclude=["tests", "tests.*", "examples", "examples.*"],
    ),
    package_dir={"": "src"},
    include_package_data=True,
    package_data=find_package_data(),
    zip_safe=False,  # Not zip safe for extension modules
    
    # Python requirements
    python_requires=PYTHON_REQUIRES,
    
    # Dependencies
    install_requires=parse_requirements()["core"],
    extras_require={
        "dev": parse_requirements()["dev"],
        "lab": parse_requirements()["lab"],
        "notebook": parse_requirements()["notebook"],
        "security": parse_requirements()["security"],
        "all": [
            *parse_requirements()["dev"],
            *parse_requirements()["lab"],
            *parse_requirements()["notebook"],
            *parse_requirements()["security"],
        ],
    },
    
    # Entry points
    entry_points=find_entry_points(),
    
    # Extensions
    ext_modules=find_extensions(),
    cmdclass={
        "build_ext": RustBuild,
    },
    
    # Data files
    data_files=[
        # Configuration files
        ("etc/synthfuse", [
            "config/cabinet.yaml.example",
            "config/opengate.yaml.example",
            "config/sigils.yaml.example",
        ]),
        
        # Documentation
        ("share/doc/synthfuse", [
            "README.md",
            "CREDITS.md",
            "LICENSE",
            "docs/Alchem-j.pdf",
            "docs/PAPER SYNTHFUSE.pdf",
        ]),
        
        # Notebook examples
        ("share/synthfuse/notebooks", [
            "notebooks/01_quickstart.ipynb",
            "notebooks/02_cabinet_tutorial.ipynb",
            "notebooks/03_sigil_composition.ipynb",
            "notebooks/04_zeta_ingestion.ipynb",
        ]),
        
        # Lab assets
        ("share/synthfuse/lab", [
            "lab/index.html",
            "lab/styles/cabinet.css",
            "lab/scripts/cabinet.js",
        ]),
    ],
    
    # Additional metadata
    project_urls={
        "Homepage": URL,
        "Documentation": "https://synth-fuse.readthedocs.io/",
        "Repository": URL,
        "Changelog": f"{URL}/releases",
        "Bug Tracker": f"{URL}/issues",
        "Discussions": f"{URL}/discussions",
    },
    
    # Options for setuptools
    options={
        "bdist_wheel": {
            "universal": False,  # Not universal (contains C/Rust extensions)
        },
        "egg_info": {
            "tag_build": "",  # No build tag for releases
        },
    },
    
    # Include all package data
    include_dirs=[],
    
    # For type checking
    ext_package="synthfuse",
    
    # Platform-specific dependencies (optional)
    platform_specific={
        "linux": ["libopenblas-dev", "liblapack-dev"],
        "darwin": ["openblas", "lapack"],
        "win32": [],
    },
    
    # Scripts (legacy, use entry_points instead)
    scripts=[],
    
    # Test suite
    test_suite="tests",
    
    # Provide eggs as zip
    zip_safe=False,
    
    # Command hooks
    setup_requires=[
        "setuptools>=68.0.0",
        "wheel>=0.42.0",
        "setuptools-rust>=1.8.0",  # For Rust extensions
    ],
)

# --------------------------------------------------------------
# Post-installation message
# --------------------------------------------------------------

def print_post_install_message():
    """Print helpful message after installation."""
    message = """
    ┌────────────────────────────────────────────────────────────┐
    │                Synth-Fuse v0.2.0 Installed!                │
    │         Unified Field Engineering - Cabinet Ready          │
    └────────────────────────────────────────────────────────────┘
    
    Quick Start:
      $ synthfuse          # Start the Cabinet
      $ sfbench --help     # View benchmarking tools
      $ sfcabinet init     # Initialize Cabinet configuration
    
    Lab Interface:
      $ synthfuse-lab      # Launch web interface
      Then open: http://localhost:8501
    
    Development:
      $ pip install -e .[dev]    # Install development dependencies
      $ pre-commit install       # Setup code quality hooks
    
    Documentation: https://synthfuse.work/docs
    Issues: https://github.com/deskiziarecords/Synth-fuse/issues
    
    Need help? Join our community discussions!
    """
    print(message)

if __name__ == "__main__":
    print_post_install_message()
