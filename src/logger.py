import os
import logging


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


def create_logger(name: str):
    if not os.path.exists("../logs/"):
        os.makedirs("../logs/")

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

    add_stream_logger(logger, log_format)
    add_file_logger(logger, log_format, '../logs/' + name + '.log')

    return logger
