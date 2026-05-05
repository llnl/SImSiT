__version__ = "1.0.0"

_base_version = __version__.split("rc", 1)[0]
__version_info__ = tuple(map(int, _base_version.split(".")))