# methods to support testing
from pkg_resources import resource_string

__all__ = []

class Context(object):
    def __init__(self):
        self._r_data = {}
        resource_string('msmbuilder', 'ProjectInfo.h5')
