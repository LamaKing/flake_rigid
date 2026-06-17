Contributing
============

Contributions to FLAKE are welcome in any form: bug reports, feature requests,
documentation improvements, or code.

Reporting bugs and requesting features
---------------------------------------

Please open an issue on the
`GitHub issue tracker <https://github.com/LamaKing/flake_rigid/issues>`_.
Include where relevant:

- FLAKE version (``pip show flake-rigid``)
- A minimal reproducible example (YAML parameter file + Python snippet or CLI command)
- The error message or unexpected output

Contributing code
-----------------

Fork the repository or open a branch, make your changes, and submit a pull
request against ``main``. A few guidelines:

- Install in editable mode with dev dependencies before starting:

  .. code-block:: console

     pip install -e ".[dev]"

- Keep new physics consistent with the conventions already in the codebase
  (units, coordinate conventions, error handling — see :doc:`intro`).
- Run the test suite and make sure nothing is broken:

  .. code-block:: console

     pytest -v -m "not slow"

- New features or substrate types are much easier to review if they come with
  a short test or example.

This is a small research project, so expectations are flexible — if in doubt,
open an issue first to discuss before investing time in a large change.

Questions
---------

For questions about the physics or the intended use of FLAKE, open a
`GitHub Discussion <https://github.com/LamaKing/flake_rigid/discussions>`_
or contact the author directly (see the repository ``CITATION.cff``).
