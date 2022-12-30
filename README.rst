========
Overview
========

.. start-badges

.. list-table::
    :stub-columns: 1

    * - docs
      - |docs|
    * - tests
      - | |github-actions| |requires|
        | |codecov|
    * - package
      - | |version| |wheel| |supported-versions| |supported-implementations|
        | |commits-since|
.. |docs| image:: https://readthedocs.org/projects/python-spipy/badge/?style=flat
    :target: https://python-spipy.readthedocs.io/
    :alt: Documentation Status

.. |github-actions| image:: https://github.com/BenjaminDoran/python-spipy/actions/workflows/github-actions.yml/badge.svg
    :alt: GitHub Actions Build Status
    :target: https://github.com/BenjaminDoran/python-spipy/actions

.. |requires| image:: https://requires.io/github/BenjaminDoran/python-spipy/requirements.svg?branch=main
    :alt: Requirements Status
    :target: https://requires.io/github/BenjaminDoran/python-spipy/requirements/?branch=main

.. |codecov| image:: https://codecov.io/gh/BenjaminDoran/python-spipy/branch/main/graphs/badge.svg?branch=main
    :alt: Coverage Status
    :target: https://codecov.io/github/BenjaminDoran/python-spipy

.. |version| image:: https://img.shields.io/pypi/v/spipy.svg
    :alt: PyPI Package latest release
    :target: https://pypi.org/project/spipy

.. |wheel| image:: https://img.shields.io/pypi/wheel/spipy.svg
    :alt: PyPI Wheel
    :target: https://pypi.org/project/spipy

.. |supported-versions| image:: https://img.shields.io/pypi/pyversions/spipy.svg
    :alt: Supported versions
    :target: https://pypi.org/project/spipy

.. |supported-implementations| image:: https://img.shields.io/pypi/implementation/spipy.svg
    :alt: Supported implementations
    :target: https://pypi.org/project/spipy

.. |commits-since| image:: https://img.shields.io/github/commits-since/BenjaminDoran/python-spipy/v0.0.1.svg
    :alt: Commits since latest release
    :target: https://github.com/BenjaminDoran/python-spipy/compare/v0.0.1...main



.. end-badges

Tools for running spectral phylogenetic inference

* Free software: BSD 3-Clause License

Installation
============

::

    pip install spipy

You can also install the in-development version with::

    pip install https://github.com/aramanlab/spipy/archive/main.zip


Documentation
=============


https://python-spipy.readthedocs.io/


Development
===========

To run all the tests run::

    tox

Note, to combine the coverage data from all the tox environments run:

.. list-table::
    :widths: 10 90
    :stub-columns: 1

    - - Windows
      - ::

            set PYTEST_ADDOPTS=--cov-append
            tox

    - - Other
      - ::

            PYTEST_ADDOPTS=--cov-append tox
