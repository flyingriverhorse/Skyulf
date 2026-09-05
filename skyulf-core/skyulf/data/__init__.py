"""Data containers and the abstract storage interface.

``dataset`` holds :class:`~skyulf.data.dataset.SplitDataset` and the
engine-neutral ``SplitPayload`` union that a split slot may contain;
``catalog`` defines the :class:`~skyulf.data.catalog.DataCatalog` ABC that
concrete storage backends implement outside the core library. Nothing is
re-exported here, so import from the submodule.
"""
