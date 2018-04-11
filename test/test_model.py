#!/usr/bin/env python 

import sys, os
sys.path.insert(0, os.path.abspath('../'))

# this project 
import physics_model 

def test_bsa():
    m = physics_model.BeamSpinAsymmetryModel() 
    print('Testing BSA Model: ', m.pars)

def test_unpol(): 
    m = physics_model.UnpolarizedSIDISModel() 
    print('Testing Unpolarized Model: ', m.pars)

if __name__ == "__main__":

    test_bsa() 
    test_unpol() 
