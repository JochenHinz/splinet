import numpy, nutils, os

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra = {}
try:
  from setuptools import setup, Extension
except:
  from distutils.core import setup, Extension
else:
  extra['install_requires'] = [ 'numpy>=1.8', 'matplotlib>=1.3', 'scipy>=0.13', 'shapely>=2.0.0' ]

long_description = """
The splinet library for Python 3, version 0.1alpha
"""

setup(
  name='splinet',
  version='0.1alpha',
  description='splinet',
  author='Jochen Hinz',
  author_email='jochen.hinz@epfl.ch',
  url='http://google.com',
  packages=[ 'splinet' ],
  long_description=long_description,
  **extra
)
