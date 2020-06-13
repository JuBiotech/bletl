import setuptools
__packagename__ = 'bletl_analysis'

def get_version():
    import os, re
    VERSIONFILE = os.path.join(__packagename__, '__init__.py')
    initfile_lines = open(VERSIONFILE, 'rt').readlines()
    VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
    for line in initfile_lines:
        mo = re.search(VSRE, line, re.M)
        if mo:
            return mo.group(1)
    raise RuntimeError('Unable to find version string in %s.' % (VERSIONFILE,))

__version__ = get_version()


setuptools.setup(name = __packagename__,
        packages = setuptools.find_packages(), # this must be the same as the name above
        version=__version__,
        description='Package for parsing and transforming BioLector raw data.',
        url='https://jugit.fz-juelich.de/IBG-1/biopro/bletl',
        download_url = 'https://jugit.fz-juelich.de/IBG-1/biopro/bletl/tarball/%s' % __version__,
        author='Michael Osthege',
        author_email='m.osthege@fz-juelich.de',
        copyright='(c) 2020 Forschungszentrum Jülich GmbH',
        license='(c) 2020 Forschungszentrum Jülich GmbH',
        classifiers= [
            'Programming Language :: Python',
            'Operating System :: OS Independent',
            'Programming Language :: Python :: 3.6',
            'Intended Audience :: Developers'
        ],
        install_requires=[
            'pandas',
            'bletl>=0.9',
            'numpy',
            'scipy',
            'joblib',
            'csaps>=0.11',
        ]
)

