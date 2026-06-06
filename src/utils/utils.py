import logging

def setup_logger(name: str, log_file: str, level=logging.INFO):
    handler = logging.FileHandler(log_file)        
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)

    return logger