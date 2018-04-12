#!/usr/bin/env python

import numpy as np
import unittest

from phifitter import generator

class TestGenerator(unittest.TestCase):

    def test_simple(self):
        phi, val, err = generator.generate() 
        self.assertTrue(len(phi) is len(val))
        self.assertTrue(len(phi) is len(err))

    def test_set_params(self):
        p = np.array([0.0, 0.0, 0.0])
        phi, val, err = generator.generate(parameters=p, error=1e-3) 

        self.assertAlmostEqual(np.sum(val), 0.0, places=1)
