.. _exampledatasets:

Example Datasets
~~~~~~~~~~~~~~~~

The examples make use of a small number of sample datasets. These are
available to view and download from a
`dedicated directory in the codebase repository <https://github.com/NCAS-CMS/cf-plot/tree/main/docs/source/example-datasets>`_.
Alternatively, they may be downloaded together as a pair of zip files
(split into two parts to fit within GitHub file size limits,
total size ~125 MB) available from
:download:`here (part 1 of 2) <https://github.com/NCAS-CMS/cf-plot/raw/main/docs/source/_downloads/example-datasets.z01>` and
:download:`here (part 2 of 2) <https://github.com/NCAS-CMS/cf-plot/raw/main/docs/source/_downloads/example-datasets.zip>` which can be
unzipped together to the full dataset directory as follows:

.. code-block:: bash
   :caption: *Accessing all of the datasets for the cf-plot examples*

   # Download the split zip parts: either use 'wget' as below or
   # download both files from the links above
   wget https://github.com/NCAS-CMS/cf-plot/tree/main/docs/source/_downloads/example-datasets.z01
   wget https://github.com/NCAS-CMS/cf-plot/tree/main/docs/source/_downloads/example-datasets.zip

   # Recombine into a single zip (zip -s0 merges split parts)
   zip -s0 example-datasets.zip --out all-example-datasets.zip

   # Unzip the merged file
   unzip all-example-datasets.zip
