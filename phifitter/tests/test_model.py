#!/usr/bin/env python 

import numpy as np 
import unittest 

from phifitter import physics_model 


class TestBSAModel(unittest.TestCase):
    def test(self):
        m = physics_model.BeamSpinAsymmetryModel() 
        self.assertTrue(m is not None)
        

class TestUnpolarizedSIDISModel(unittest.TestCase):
    def test(self): 
        m = physics_model.UnpolarizedSIDISModel() 
        self.assertTrue(m is not None)

if __name__ == "__main__":
    unittest.main() 
