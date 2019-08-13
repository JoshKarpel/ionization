from pathlib import Path

from setuptools import setup, find_packages

THIS_DIR = Path(__file__).parent


setup(
    name="ionization",
    version="0.1.0",
    author="Josh Karpel",
    author_email="josh.karpel@gmail.com",
    description="Ionization simulations for the Yavuz Group.",
    long_description=Path("README.md").read_text(),
    url="https://github.com/JoshKarpel/ionization",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Physics",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: System :: Distributed Computing",
    ],
    packages=find_packages(exclude=["dev", "docker", "science", "tests"]),
    package_data={"": ["*.pyx"]},
    entry_points={
        "console_scripts": [
            f"{x.stem}=ionization_scans.{x.stem}:main"
            for x in (THIS_DIR / "ionization_scans").iterdir()
            if x.stem.startswith(("scan", "export"))
        ]
    },
    install_requires=Path("requirements.txt").read_text().splitlines(),
)
