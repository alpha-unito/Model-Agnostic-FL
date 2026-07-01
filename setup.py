# Copyright (C) 2020-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import glob

from setuptools import Command
from setuptools import setup
from setuptools.command.build import build  # setuptools >= 62.4

try:
    # Legacy editable installs (pip install -e . with no pyproject) use `develop`.
    from setuptools.command.develop import develop
except ImportError:  # pragma: no cover
    develop = None


def _read_long_description():
    try:
        with open("README.md", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""


def _generate_protos():
    """Generate *_pb2*.py from every .proto under the repo root."""
    try:
        from grpc_tools import command
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "grpcio-tools is required to build openfl-x's gRPC stubs.\n"
            "Install it and build without isolation:\n"
            "    pip install 'grpcio-tools~=1.73.0'\n"
            "    pip install . --no-build-isolation"
        ) from exc
    command.build_package_protos(".")


class BuildGrpc(Command):
    """Generate gRPC/protobuf Python stubs from .proto files."""

    description = "generate gRPC *_pb2*.py stubs from openfl/protocols/*.proto"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        _generate_protos()

    def get_source_files(self):
        return glob.glob("openfl/protocols/*.proto")

    def get_outputs(self):
        return []

    def get_output_mapping(self):
        return {}


class BuildWithGrpc(build):
    """Standard build with proto generation prepended as a sub-command."""

    sub_commands = [("build_grpc", None)] + build.sub_commands


cmdclass = {
    "build": BuildWithGrpc,
    "build_grpc": BuildGrpc,
}

if develop is not None:
    class DevelopGrpc(develop):
        """Legacy editable install path: generate stubs before linking."""

        def run(self):
            self.run_command("build_grpc")
            super().run()

    cmdclass["develop"] = DevelopGrpc


setup(
    name="openfl-x",
    version="1.4.0.dev4",
    author="Gianluca Mittone",
    description="Model-agnostic federated learning",
    long_description=_read_long_description(),
    long_description_content_type="text/markdown",
    url="https://github.com/alpha-unito/OpenFL-extended",
    license="Apache-2.0",
    python_requires=">=3.9,<3.14",  # was >=3.6, <3.11
    packages=[
        "openfl",
        "openfl.component",
        "openfl.interface.aggregation_functions",
        "openfl.interface.aggregation_functions.core",
        "openfl.interface.aggregation_functions.experimental",
        "openfl.component.aggregator",
        "openfl.component.assigner",
        "openfl.component.ca",
        "openfl.component.collaborator",
        "openfl.component.director",
        "openfl.component.envoy",
        "openfl.component.straggler_handling_functions",
        "openfl.cryptography",
        "openfl.databases",
        "openfl.databases.utilities",
        "openfl.federated",
        "openfl.federated.data",
        "openfl.federated.plan",
        "openfl.federated.task",
        "openfl.interface",
        "openfl.interface.interactive_api",
        "openfl.native",
        "openfl.pipelines",
        "openfl.plugins",
        "openfl.plugins.frameworks_adapters",
        "openfl.plugins.interface_serializer",
        "openfl.plugins.processing_units_monitor",
        "openfl.protocols",
        "openfl.transport",
        "openfl.transport.grpc",
        "openfl.utilities",
        "openfl.utilities.data_splitters",
        "openfl.utilities.fedcurv",
        "openfl.utilities.fedcurv.torch",
        "openfl.utilities.optimizers.keras",
        "openfl.utilities.optimizers.numpy",
        "openfl.utilities.optimizers.torch",
        "openfl-docker",
        "openfl-gramine",
        "openfl-tutorials",
        "openfl-workspace",
    ],
    include_package_data=True,
    install_requires=[
        "Click>=8.1,<9",          # was ==8.0.1
        "PyYAML>=6.0",            # was >=5.4.1
        "cloudpickle",
        "cryptography>=42.0",     # was >=3.4.6
        "dill",
        "docker",
        "dynaconf>=3.2,<4",       # was ==3.1.5
        "flatten_json",
        "grpcio>=1.73,<2.0",      # was ~=1.48.2 (runtime)
        "protobuf>=5.28,<7",      # was ==3.19.5 (must match grpcio-tools gencode; <7 required)
        "ipykernel",
        "jupyterlab",
        "numpy",                  # pin numpy<2 if you hit AttributeError/dtype errors on import
        "pandas",
        "requests",
        "rich",
        "scikit-learn",
        "tensorboard",
        "tensorboardX",
        "tqdm",
        "wandb",
    ],
    # Best-effort only; see BUILD DEPENDENCY NOTE at top of file.
    setup_requires=["grpcio-tools~=1.73.0"],
    entry_points={
        "console_scripts": ["fx=openfl.interface.cli:entry"],
    },
    project_urls={
        "Source Code": "https://github.com/alpha-unito/OpenFL-extended",
    },
    classifiers=[
        "Environment :: Console",
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: System :: Distributed Computing",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    cmdclass=cmdclass,
)