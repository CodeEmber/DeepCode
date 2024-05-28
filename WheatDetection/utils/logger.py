import logging.config
from utils.file_utils import get_file_path
import inspect


class LoggerManager():

    def __init__(self) -> None:
        logger_path = get_file_path(path=["logging.ini"])
        logging.config.fileConfig(logger_path)
        self.logger = logging.getLogger("test")

    def log(self, level, message):
        frame_info = inspect.stack()[2]
        file_name = frame_info.filename
        file_name = file_name.split('/')[-2] + '/' + file_name.split('/')[-1]
        line_number = frame_info.lineno
        func_name = frame_info.function
        log_message = f"[{file_name}:{line_number}:{func_name}] {message} "
        self.logger.log(logging.getLevelName(level.upper()), log_message)

    def info(self, message):
        self.log('info', message)

    def debug(self, message):
        self.log('debug', message)

    def warning(self, message):
        self.log('warning', message)

    def success(self, message):
        self.log('success', message)

    def error(self, message):
        self.log('error', message)


logger = LoggerManager()
