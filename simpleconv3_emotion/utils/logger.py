'''
Author       : wyx-hhhh
Date         : 2023-04-29
LastEditTime : 2023-04-29
Description  : 日志管理
'''
import logging.config


class MyLogger:

    def __init__(self):
        logging.config.fileConfig('logging.ini')
        self.logger = logging.getLogger('test')


if __name__ == '__main__':
    logger = MyLogger()
    logger.logger.debug('debug message')
    logger.logger.info('info message')
    logger.logger.warning('warning message')
    logger.logger.error('error message')
