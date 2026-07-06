"""Data-loading utilities for MSE806 Assessment 2.

Two artefacts live here:

* :pyfunc:`mock_data_loader.build_mock_dataset` – synthesises a
  ``(T, N, F)`` tensor of traffic speed that mimics METR-LA in shape and
  statistical profile.  It is used everywhere in the repository so that
  the pipeline can be exercised without downloading the real dataset.
* :pyfunc:`preprocessing.build_adjacency_matrix` – computes the
  thresholded Gaussian-kernel adjacency matrix from a pairwise distance
  matrix, following Li et al. (2018).

Both modules are deliberately lightweight so they can be imported without
PyTorch (only NumPy / SciPy).
"""
