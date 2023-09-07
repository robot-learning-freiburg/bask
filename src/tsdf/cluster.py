try:
    # TODO: configure the path somewhere central
    import os
    os.environ["CUDA_PATH"] = 'PATH'
    os.environ["LD_LIBRARY_PATH"] = 'PATH'
    from cuml.cluster import DBSCAN as Cluster  # type: ignore

except Exception as err:
    from loguru import logger
    logger.warning('{}', err)
    logger.warning('Failed to import cuml. Running clustering in CPU mode.')
    from sklearn.cluster import DBSCAN as Cluster  # type: ignore
