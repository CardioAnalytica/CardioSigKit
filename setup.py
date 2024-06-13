# -*- coding: utf-8 -*-
#
"""
Main setup for the library.
Supports Python >= 3.8.
Usage as usual with setuptools:
    python setup.py build_ext
    python setup.py build
    python setup.py install
    python setup.py sdist
For details, see
    http://setuptools.readthedocs.io/en/latest/setuptools.html#command-reference
or
    python setup.py --help
    python setup.py --help-commands
    python setup.py --help bdist_wheel  # or any command
"""

#########################################################
# General config
#########################################################

import os

libname = "cardiosigkit"
libdir = os.path.dirname(os.path.realpath(__file__))
build_type="optimized"
#build_type="debug"

SHORTDESC="cardiosigkit"
DESC="""Comming soon long description
"""

datadirs  = ("extra",)
dataexts  = (".py",  ".pyx", ".pxd",  ".c", ".cpp", ".h",  ".sh",  ".lyx", ".tex", ".txt", ".pdf")
standard_docs     = ["README", "LICENSE", "TODO", "CHANGELOG", "AUTHORS"]
standard_doc_exts = [".md", ".rst", ".txt", ""]

print()

#########################################################
# Init
#########################################################


import sys
if sys.version_info < (3,6):
    sys.exit('Sorry, Python < 3.6 is not supported')

from setuptools import setup
from setuptools.extension import Extension
import numpy
import multiprocessing
import platform

try:
    from Cython.Build import cythonize
except ImportError:
    sys.exit("Cython not found. Cython is needed to build the extension modules.")


#########################################################
# Definitions
#########################################################


# Modules involving numerical computations
#
if platform.system() != "Darwin" and "arm" not in platform.platform():
    extra_compile_args_math_optimized = ['-march=native', '-O2', '-msse', '-msse2', '-mfma', '-mfpmath=sse']
else:
    print("Apple Silicon on MacOS architecture detected")
    extra_compile_args_math_optimized = ['-march=native', '-O2', '-mfma']
extra_compile_args_math_debug        = ['-march=native', '-O0', '-g']
extra_link_args_math_optimized       = []
extra_link_args_math_debug           = []

# Modules that do not involve numerical computations
#
extra_compile_args_nonmath_optimized = ['-O2']
extra_compile_args_nonmath_debug     = ['-O0', '-g']
extra_link_args_nonmath_optimized    = []
extra_link_args_nonmath_debug        = []

# Additional flags to compile/link with OpenMP
#
if platform.system() != "Darwin":
    openmp_compile_args = ['-fopenmp']
    openmp_link_args    = ['-fopenmp']
else:
    openmp_compile_args = []
    openmp_link_args = []

# CPU count to run multiple threads for Cythonize
#
jobs = multiprocessing.cpu_count()


#########################################################
# Helpers
#########################################################


default_include_dirs = [".", "./cardiosigkit"]
# default_include_dirs = [".", "./cardiosigkit", "/data/projects/deepecg/libs/jupyter"]
# default_include_dirs = [".", "./cardiosigkit", "/data/projects/deepecg/libs/sciai"]

if build_type == 'optimized':
    my_extra_compile_args_math    = extra_compile_args_math_optimized
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_optimized
    my_extra_link_args_math       = extra_link_args_math_optimized
    my_extra_link_args_nonmath    = extra_link_args_nonmath_optimized
    debug = False
    print( "build configuration selected: optimized" )
elif build_type == 'debug':
    my_extra_compile_args_math    = extra_compile_args_math_debug
    my_extra_compile_args_nonmath = extra_compile_args_nonmath_debug
    my_extra_link_args_math       = extra_link_args_math_debug
    my_extra_link_args_nonmath    = extra_link_args_nonmath_debug
    debug = True
    print( "build configuration selected: debug" )
else:
    raise ValueError("Unknown build configuration '%s'; valid: 'optimized', 'debug'" % (build_type))


def declare_cython_extension(extName, extFiles=None, use_math=False, use_openmp=False, include_dirs=None):
    """
    Declare a Cython extension module for setuptools.
    @param extName: str. Absolute module name with dotted style notation.
    @param extFiles: array of str or str. Absolute source files to be compiled with the extension, written in dotted style notation.
    @param use_math: bool. If True, set math flags and link with ``ibm``.
    @param use_openmp: bool. If True, compile and link with OpenMP.
    @param include_dirs: array of str. Directories to include during compilation.
    @return: Extension object that can be passed to ``setuptools.setup``.
    """

    if extFiles is None:
        extFiles = extName + ".pyx"
    if type(extFiles) not in (list, tuple):
        extFiles = [extFiles.replace(".pyx", "").replace(".", os.path.sep)+".pyx"]
    else:
        extFiles = [extFile.replace(".pyx", "").replace(".", os.path.sep) + ".pyx" for extFile in extFiles]

    if use_math:
        compile_args = list(my_extra_compile_args_math) # copy
        link_args    = list(my_extra_link_args_math)
        libraries    = ["m"] if platform.system() != "Windows" else [] # link libm; this is a list of library names without the "lib" prefix
    else:
        compile_args = list(my_extra_compile_args_nonmath)
        link_args    = list(my_extra_link_args_nonmath)
        libraries    = None  # value if no libraries, see setuptools.extension._Extension

    # OpenMP
    if use_openmp:
        compile_args = openmp_compile_args + compile_args
        link_args = openmp_link_args + link_args

    # Specify C++ version
    compile_args = [*compile_args, "-std=c++14"]
    link_args = [*link_args, "-std=c++14"]

    if include_dirs is not None:
        _include_dirs = [*default_include_dirs, *include_dirs]
    else:
        _include_dirs = default_include_dirs

    return Extension(extName,
                     extFiles,
                     extra_compile_args=compile_args,
                     extra_link_args=link_args,
                     include_dirs=_include_dirs,
                     libraries=libraries,
                     language="c++")


# Gather extra data files
#
datafiles = []
getext = lambda filename: os.path.splitext(filename)[1]
for datadir in datadirs:
    datafiles.extend( [(root, [os.path.join(root, f) for f in files if getext(f) in dataexts])
                       for root, dirs, files in os.walk(datadir)] )


# Add standard documentation (README et al.), if any, to data files
#
detected_docs = []
for docname in standard_docs:
    for ext in standard_doc_exts:
        filename = "".join((docname, ext))
        if os.path.isfile(filename):
            detected_docs.append(filename)
datafiles.append(('.', detected_docs))


# Extract __version__ from the package __init__.py
#
import ast
init_py_path = os.path.join(".", '__init__.py')
version = '0.0.unknown'
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith('__version__'):
                version = ast.parse(line).body[0].value.s
                break
        else:
            print("WARNING: Version information not found in '%s', using placeholder '%s'" % (init_py_path, version), file=sys.stderr)
except FileNotFoundError:
    print("WARNING: Could not find file '%s', using placeholder version information '%s'" % (init_py_path, version), file=sys.stderr)


#########################################################
# Set up modules
#########################################################

# declare Cython extension modules in logical order
#
cython_ext_modules = [
    declare_cython_extension("cardiosigkit.helpers.xmlutils", use_math=True, use_openmp=True, include_dirs=[numpy.get_include()]),
    declare_cython_extension("cardiosigkit.helpers.serialization", use_math=True, use_openmp=True, include_dirs=[numpy.get_include()]),
    declare_cython_extension("cardiosigkit.helpers.nested_map", use_math=True, use_openmp=True, include_dirs=[numpy.get_include()]),
    declare_cython_extension("cardiosigkit.parsers.parser", use_math=True, use_openmp=True, include_dirs=[numpy.get_include()]),
    declare_cython_extension("cardiosigkit.parsers.hl7_a_ecg.hl7_a_ecg", use_math=True, use_openmp=True, include_dirs=[numpy.get_include()]),
    declare_cython_extension("cardiosigkit.parsers.philips_sierra_xml.sierra_xml", use_math=True, use_openmp=True, include_dirs=[numpy.get_include()]),
]

# Call cythonize() explicitly, as recommended in the Cython documentation. See
#     http://cython.readthedocs.io/en/latest/src/reference/compilation.html#compiling-with-distutils
#
# This will favor Cython's own handling of '.pyx' sources over that provided by setuptools.
#
# Disabling multiprocessing if compiling on windows
if platform.system() in ("Windows", "Darwin"):
    ext_modules = cythonize(cython_ext_modules, include_path=default_include_dirs, gdb_debug=debug, compiler_directives={'language_level' : "3", 'unraisable_tracebacks': False})
else:
    ext_modules = cythonize(cython_ext_modules, include_path=default_include_dirs, gdb_debug=debug,
                            compiler_directives={'language_level': "3", 'unraisable_tracebacks': False}, nthreads=jobs)

#########################################################
# Set up packages
#########################################################

base_dir = os.path.join(libdir, libname)
base_entries = os.walk(base_dir)
packages = [libname]
for entry in base_entries:
    if os.path.relpath(entry[0], base_dir) == ".":
        continue
    if "__init__.py" in entry[2]:
        packages.append(libname+"."+os.path.relpath(entry[0], base_dir).replace(os.path.sep, "."))

packages_data = {}
for package in packages:
    packages_data[package] = ['*.pxd']

#########################################################
# Call setup()
#########################################################

setup(
    name = "cardiosigkit",
    version = version,
    author = "Ahmad Fall",
    author_email = "ahmad.fall@ird.fr",
    url = "https://git.ummisco.fr/deepomics/cardiosigkit",
    description = SHORTDESC,
    long_description = DESC,
    license = "Unlicense",
    platforms = ["Linux"],
    classifiers = ["Development Status :: 3 - Snapshot 0",
                   "Environment :: Console",
                   "Intended Audience :: Developers",
                   "Intended Audience :: Science/Research",
                   "License :: Unlicense",
                   "Operating System :: POSIX :: Linux",
                   "Programming Language :: Cython",
                   "Programming Language :: Python",
                   "Programming Language :: Python :: 3",
                   "Programming Language :: Python :: 3.8",
                   "Topic :: Scientific/Engineering",
                   "Topic :: Scientific/Engineering :: Mathematics",
                   "Topic :: Scientific/Engineering :: Artificial Intelligence",
                   "Topic :: Scientific/Engineering :: Deep Learning",
                   "Topic :: Software Development :: Libraries",
                   "Topic :: Software Development :: Libraries :: Python Modules"],
    setup_requires = ["cython", "numpy"],
    install_requires = ["numpy", "wurlitzer"],
    provides = ["cardiosigkit"],
    keywords = ["cardiosigkit ecg mongo db database datastore dataset"],
    ext_modules = ext_modules,
    packages = packages,
    package_data=packages_data,
    zip_safe = False,
    data_files = datafiles
)