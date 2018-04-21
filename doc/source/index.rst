========================================
feets: feATURE eXTRACTOR FOR tIME sERIES
========================================

.. only:: html

    .. image:: _static/logo_medium.png
        :align: center
        :scale: 100 %

In time-domain astronomy, data gathered from the telescopes is usually
represented in the form of light-curves. These are time series that show the
brightness variation of an object through a period of time
(for a visual representation see video below). Based on the variability
characteristics of the light-curves, celestial objects can be classified into
different groups (quasars, long period variables, eclipsing binaries, etc.)
and consequently be studied in depth independentely.
In order to characterize this variability, some of the existing methods use
machine learning algorithms that build their decision on the light-curves
features. Features, the topic of the following work, are numerical descriptors
that aim to characterize and distinguish the different variability classes.
They can go from basic statistical measures such as the mean or the standard
deviation, to complex time-series characteristics such as the autocorrelation
function.
In this package we present a library with a compilation of some of the
existing light-curve features. The main goal is to create a collaborative and
open tool where every user can characterize or analyze an astronomical
photometric database while also contributing to the library by adding new
features. However, it is important to highlight that **this library is not**
**restricted to the astronomical field** and could also be applied to any kind
of time series.
Our vision is to be capable of analyzing and comparing light-curves from all
the available astronomical catalogs in a standard and universal way. This
would facilitate and make more efficient tasks as modelling, classification,
data cleaning, outlier detection and data analysis in general. Consequently,
when studying light-curves, astronomers and data analysts would be on the same
wavelength and would not have the necessity to find a way of comparing or
matching different features. In order to achieve this goal, the library should
be run in every existent survey (MACHO, EROS, OGLE, Catalina, Pan-STARRS, etc)
and future surveys (LSST) and the results should be ideally shared in the same
open way as this library.

Help & discussion mailing list
------------------------------

.. ~ Our Google Groups mailing list is
.. ~ `here <https://groups.google.com/forum/#!forum/scikit-criteria>`_.


**You can contact me at:** jbc.develop@gmail.com (if you have a support
question, try the mailing list first)


Code Repository & Issues
------------------------

https://github.com/carpyncho/feets


License
-------

feets is under
`The MIT License <https://raw.githubusercontent.com/carpyncho/feets/master/LICENSE>`__

A short and simple permissive license with conditions only requiring preservation of copyright and license notices. Licensed works, modifications, and larger works may be distributed under different terms and without source code.


Citation
--------

If you use feets in a scientific publication, we would appreciate
citations to the following paper:

    Cabral, Juan B.,


Bibtex entry::

    WORKING ON IT

**Full Publication:** FOO

`Working paper <https://github.com/carpyncho/feets_paper>`_


Contents
--------

.. toctree::
    :maxdepth: 2

    install
    tutorial.ipynb
    api/modules.rst


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Code of conduct
---------------

feets endorse
`the Astropy Project code of conduct <http://www.astropy.org/code_of_conduct.html>`_.
