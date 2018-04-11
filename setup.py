from setuptools import setup

setup(name             = 'phifitter',
      version          = '0.1',
      description      = 'Package for fitting phi distributions',
      url              = 'http://github.com/dmriser/phifitter',
      author           = 'David Riser',
      author_email     = 'dmriser@gmail.com',
      license          = 'MIT',
      packages         = ['phifitter'],
      install_requires = [
          'numpy',
          'matplotlib',
          'scipy',
          'tqdm',
          'vegas'
      ],
      test_suite = 'phifitter.tests.get_suite',
      zip_safe = False
      )
