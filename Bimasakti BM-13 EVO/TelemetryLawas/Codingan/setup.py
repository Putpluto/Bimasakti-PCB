from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules=cythonize("demoimgui_copy_4.py", compiler_directives={"language_level": "3"})
)
