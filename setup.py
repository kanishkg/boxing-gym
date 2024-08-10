from setuptools import setup, find_packages

# Function to read the requirements.txt file
def parse_requirements(filename):
    with open(filename, 'r') as file:
        return file.read().splitlines()

setup(
    name="boxing_gym",  # Name of your package
    version="0.1.0",  # Initial release version
    packages=find_packages(where='src'),  # Include all packages under src
    package_dir={"": "src"},  # Tell setuptools packages are under src
    include_package_data=True,  
    install_requires=parse_requirements('requirements.txt'),  # Parse requirements.txt for dependencies
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',  # Specify the Python version requirement
)
