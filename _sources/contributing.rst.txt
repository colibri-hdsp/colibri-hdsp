How to Contribute
=================


Colibri operates as a community-driven initiative and eagerly welcomes contributions in various capacities. 
Our ultimate goal is to cultivate a comprehensive repository encompassing optical computational imaging design 
and deep learning, and your involvement is crucial to realizing this objective.

We extend gratitude to all contributors, recognizing their efforts both in the documentation and
within the source code. Particularly noteworthy contributions will also be duly considered when 
determining authorship for forthcoming publications.

The preferred way to contribute to ``colibri`` is to fork the `main
repository <https://github.com/colibri-hdsp/colibri-hdsp>`_ on GitHub,
then submit a "Pull Request" (PR). When preparing the PR, please make sure to
check the following points:


- Validate that the automated tests pass on your local machine.  This can be accomplished by executing ``python -m pytest tests`` within the root directory of the repository post implementing the desired modifications.
- Update the documentation as necessary. Post modification, this can be executed within the "docs" directory utilizing one of the commands listed in the table below:

.. list-table::
   :widths: 40 50
   :header-rows: 1

   * - Command
     - Description of command
   * - ``make html``
     - Generates all the documentation
   * - ``make html-noplot``
     - Generates documentation faster but without plots

For those unfamiliar with the GitHub contribution process, an alternative option is to initiate an issue via the 
`issue tracker <https://github.com/colibri-hdsp/colibri-hdsp/issues>`_. We commit to promptly addressing any raised issues. 
Alternatively, you may directly communicate your inquiries or suggestions by emailing any of the principal developers.