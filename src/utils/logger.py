import logging


def get_logger(name="eci_logger"):
    logger = logging.getLogger(name)
    logging.basicConfig(
        format="%(asctime)s - %(filename)s - %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
        level=logging.INFO
    )

    return logger


if __name__ == "__main__":
    logger = get_logger()
    logger.info("test")
