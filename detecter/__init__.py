import logging

logger = logging.getLogger("detecter")
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler("log/detector.log", mode="a+")
fh.setLevel(logging.DEBUG)
fh.setFormatter(logging.Formatter("[%(asctime)s:%(levelname)s] - %(message)s"))

stderr = logging.StreamHandler()
stderr.setLevel(logging.INFO)

logger.addHandler(fh)
logger.addHandler(stderr)

