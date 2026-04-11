from setuptools import setup, find_packages

setup(
    name="openenv-email_env",
    version="0.3.0",
    packages=["email_env", "email_env.server"],
    package_dir={"email_env": "."},
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
