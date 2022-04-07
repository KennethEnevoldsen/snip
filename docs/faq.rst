FAQ
-------


How do I test the code and run the test suite?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

This package comes with a test suite. To run the tests, you should clone the repository,
Then install the package. This will also install the required development dependencies
and test utilities defined in the requirements.txt.


.. code-block::
   
   pip install -r requirements.txt
   pirp install -editable .

   python -m pytest


which will run all the test in the :code:`tests` folder.

Specific tests can be run using:

.. code-block::

   python -m pytest tests/desired_test.py


If you want to check code coverage you can run the following:

.. code-block::

   pip install pytest-cov

   python -m pytest --cov=.


Does the package run on X?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The package is intended to run on all major OS, this includes Windows (latest version),
MacOS (latest version) and the latest version of Ubuntu (Linux). 
Similarly it also tested on python 3.7, 3.8, and 3.9.

Please note these are only the systems the package is being actively tested on, if you
run on a similar system (e.g. an earlier version of Linux) the package
will likely run there as well, if not please create an issue.

How is the documentation generated?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Augmenty uses `sphinx <https://www.sphinx-doc.org/en/master/index.html>`__ to generate
documentation. It uses the `Furo <https://github.com/pradyunsg/furo>`__ theme with
custom styling.

To make the documentation you can run:

.. code-block::

  # install sphinx, themes and extensions
  pip install -r requirements.txt

  # generate html from documentations
  make -C docs html


How do I cite this work?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
If you use this library in your research, it would be much appreciated it if you would cite:

.. code-block::
   
   @inproceedings{name2022,
      title={Name, very catchy subtitle},
      author={Enevoldsen, Kenneth},
      year={2022}
   }
