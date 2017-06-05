'''
Base Class for all Environments
'''


class BaseEnv(object):
    def init_env(self):
        pass

    def start(self):
        pass

    def step(self, action):
        pass

    def reset(self):
        pass
