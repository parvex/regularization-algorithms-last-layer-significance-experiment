import logging
import os
import wandb


class Logger:
    def __init__(self, log_file):
        self.log_file = log_file
        logging.basicConfig(filename=log_file, level=logging.INFO)


    def log(self, message):
        print(message)
        logging.info(message)
