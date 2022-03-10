import os
import logging


def create_logger(log_file: str):
    def add_stream_logger(logger, log_format: str):
        handler = logging.StreamHandler()
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    def add_file_logger(logger, log_format: str, name: str):
        handler = logging.FileHandler(name)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter(log_format)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    directory = os.path.dirname(log_file)
    if not os.path.exists(directory):
        os.makedirs(directory)

    logger = logging.getLogger('Experiment')
    logger.setLevel(logging.DEBUG)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    add_stream_logger(logger, log_format)
    add_file_logger(logger, log_format, log_file)

    return logger
