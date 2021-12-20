import logging


class Logger:
    def __init__(self, log_path, to_stdout=False):
        logging.basicConfig(filename=log_path, level=logging.INFO)

        if to_stdout:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            logging.getLogger().addHandler(ch)

        self.logger = logging.getLogger()


    def log(self, content, name=None):
        if name is None:
            self.logger.info("%s" % (content))
        else:
            self.logger.info("%s: %s" % (name, content))

