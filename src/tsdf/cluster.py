try:
    import os
    os.environ["CUDA_PATH"] = '/usr/local/cuda-11.1'
    os.environ["LD_LIBRARY_PATH"] = '/usr/local/cuda-11.1/lib64:/home/PLEASE_ADAPT_USERNAME/CoppeliaSim_Edu_V4_1_0_Ubuntu18_04'
    from cuml.cluster import DBSCAN as Cluster

except Exception as err:
    from loguru import logger
    logger.warning('{}', err)
    logger.warning('Failed to import cuml. Running clustering in CPU mode.')
    from sklearn.cluster import DBSCAN as Cluster  # , OPTICS  # noqa 401

# cluster = DBSCAN(eps=0.02, min_samples=1000)
# cluster = OPTICS(min_samples=1000, n_jobs=6)  # try max_eps=0.05
