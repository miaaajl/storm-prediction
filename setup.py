try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


setup(name='forecaster',
      use_scm_version=True,
      setup_requires=['setuptools_scm'],
      version='1.0',
      description='Weather forecasting',
      author='Lilian',
      packages=['forecaster']
      )
