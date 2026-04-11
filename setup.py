from setuptools import setup, find_packages

setup(
    name="openenv-email_env",
    version="0.3.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "openenv-core[core]>=0.2.1",
        "openai",
        "requests",
    ],
    extras_require={
        "dev": [
            "pytest>=8.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
)
