"""Force platform-specific wheel tag so bundled libariaengine_ffi is shipped correctly."""
from setuptools import setup
from setuptools.dist import Distribution


class BinaryDistribution(Distribution):
    """Mark this distribution as containing binary extensions (bundled .so/.dylib/.dll)."""

    def has_ext_modules(self):
        return True


setup(distclass=BinaryDistribution)
