#!/usr/bin/env python3

"""
Runs all tests. 
The .py extension is needed so this will work on windows?
"""

import unittest

if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover("test")
    unittest.TextTestRunner(verbosity=2).run(suite)
