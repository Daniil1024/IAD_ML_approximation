# https://notabug.org/scossu/lsup_rdf/src/e08da1a83647454e98fdb72f7174ee99f9b8297c
from glob import glob
from os import path
from setuptools import Extension, setup


ROOT_DIR = path.dirname(path.realpath(__file__))
MOD_DIR = path.join(ROOT_DIR, 'cpython')
SRC_DIR = path.join(ROOT_DIR, 'src')
INCL_DIR = path.join(ROOT_DIR, 'include')
EXT_DIR = path.join(ROOT_DIR, 'ext')

sources = (
    glob(path.join(SRC_DIR, '*.c')) +
    glob(path.join(MOD_DIR, '*.c')) +
    [
        path.join(EXT_DIR, 'xxHash', 'xxhash.c'),
        path.join(EXT_DIR, 'openldap', 'libraries', 'liblmdb', 'mdb.c'),
        path.join(EXT_DIR, 'openldap', 'libraries', 'liblmdb', 'midl.c'),
    ]
)

compile_args = ['-std=c99', '-DDEBUG', '-g3']


setup(
    name="lsup_rdf",
    version="1.0a1",
    description='Ultra-compact RDF library.',
    author='Stefano Cossu <https://notabug.org/scossu>',
    url='https://notabug.org/scossu/lsup_rdf',
    license='https://notabug.org/scossu/lsup_rdf/src/master/LICENSE',
    package_dir={'lsup_rdf': path.join(MOD_DIR, 'lsup_rdf')},
    packages=['lsup_rdf'],
    ext_modules=[
        Extension(
            "_lsup_rdf",
            sources,
            include_dirs=[INCL_DIR],
            libraries=['uuid'],
            extra_compile_args=compile_args,
        ),
    ],
)

