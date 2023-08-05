import tqdm
from loguru import logger


class DuplicateFilter():
    def __init__(self):
        self.msgs = set()

    def __call__(self, record):
        unseen = (msg := record["message"]) not in self.msgs
        self.msgs.add(msg)
        return unseen or not record['extra'].get('filter', True)


def setup_logger():
    logger.remove()
    duplicate_filter = DuplicateFilter()
    log_format = '<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> |' \
        ' <level>{level: <8}</level> |  <level>{message}</level>'
    logger.add(lambda msg: tqdm.tqdm.write(msg, end=""), colorize=True,
               format=log_format, filter=duplicate_filter)


setup_logger()
