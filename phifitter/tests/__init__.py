import unittest

def get_suite():
    " Return a unittest.TestSuite. "
    import phifitter.tests
    
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    return suite
