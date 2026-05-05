===============
Developer Guide
===============

In order to help develop SImSiT, the library should be installed from source.

.. code-block:: console

    git clone https://github.com/LLNL/simsit.git

Working in the Dev Container
----------------------------

For convenience, we include a Dockerfile for development within the container.
Instructions on installing Docker and its basic usage can be found on the `Docker Website <https://www.docker.com/>`_.

To build the container:

.. code-block:: console

    cd simsit
    docker build ./ -t simsit:latest

This will build the Docker image.
To check that it has built correctly, run the test suite:

.. code-block:: console

    docker run -it -v "./:/simsit" simsit:latest bash -c "cd simsit/ && pytest -v"
