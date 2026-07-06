"""Model implementations for the MSE806 A2 reference project.

Two families of models are exposed:

* :class:`models.dcrnn.DCRNN` – the graph-based Diffusion Convolutional
  Recurrent Neural Network of Li et al. (2018) with encoder / decoder
  and scheduled sampling.
* :class:`models.baselines.HistoricalAverage` and
  :class:`models.baselines.LSTMBaseline` – simple but well-defined
  baselines used in the accompanying report tables.
"""
