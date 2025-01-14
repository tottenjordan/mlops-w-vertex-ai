
from setuptools import find_packages
from setuptools import setup
import setuptools

from distutils.command.build import build as _build
import subprocess

REQUIRED_PACKAGES = [
    'google-cloud-aiplatform[cloud_profiler]>=1.71.0',
    # 'tensorflow==2.9.3',
    'tensorflow==2.13.0',
    'protobuf>=3.9.2,<3.20',
    
]

setup(
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    name='tf_trainer',
    version='0.1',
    url="wwww.google.com",
    description='Vertex AI | Training | Python Package'
)
