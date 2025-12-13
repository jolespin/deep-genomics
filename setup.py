from setuptools import setup, find_packages
from pathlib import Path

# Read long description from README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

pypi_name = "torch-genomics"
package_name = "torchgenomics"

# Read version
version = None
with open(this_directory / package_name / '__init__.py') as f:
    for line in f.readlines():
        line = line.strip()
        if line.startswith("__version__"):
            version = line.split("=")[-1].strip().strip('"')
assert version is not None, f"Check version in {package_name}/__init__.py"

# Read requirements
requirements = []
with open(this_directory / 'requirements.txt') as f:
    for line in f.readlines():
        line = line.strip()
        if line and not line.startswith("#"):
            requirements.append(line)

setup(
    name=pypi_name,
    version=version,
    description='Deep learning utilities for genomics with PyTorch',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url=f'https://github.com/jolespin/{pypi_name}',
    author='Josh L. Espinoza',
    author_email='jol.espinoz@gmail.com',
    license='MIT',
    packages=find_packages(exclude=['tests', 'notebooks', 'docs']),
    install_requires=requirements,
    python_requires='>=3.8',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Programming Language :: Python :: 3.13',
    ],
    keywords='genomics, bioinformatics, deep learning, pytorch, VAE, machine learning',
    include_package_data=False,
)
