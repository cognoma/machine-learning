import setuptools

setuptools.setup(
    name='cognoml',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.1.0',

    description='Machine learning for Project Cognoma',

    # The project's main homepage.
    url='https://github.com/cognoma/machine-learning',

    # Author details
    author='Project Cognoma',

    # Choose your license
    license='BSD 3-Clause',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        'Development Status :: 3 - Alpha',

        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: BSD License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.5',
    ],

    # What does your project relate to?
    keywords='cognoma machine learning cancer',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['cognoml'],
)
