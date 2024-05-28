'''
Author       : wyx-hhhh
Date         : 2023-04-29
LastEditTime : 2023-09-24
Description  : 获取配置
'''
import yaml


class Config:

    def __init__(self, config_file=None):
        if config_file is None:
            config_file = '/Users/wyx/程序/本科/机器学习/simpleconv3_emotion/configs/config.yaml'
        with open(config_file, 'r') as f:
            self.config = yaml.load(f, Loader=yaml.FullLoader)

    def get(self, key):
        return self.config.get(key, None)


if __name__ == "__main__":
    config = Config()
    print(config.get('device'))