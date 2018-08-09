"""
Unit test wrapper
Just imporing the test packages and running this as a script does the trick
"""


import unittest
import astropy.io.fits as fits
import matplotlib.pyplot as plt

#from Test.test_combine import TestCombineMethods
from Test.test_extract import TestExtractMethods

def compare_idl_python():
    fname_idl = "./Test/UVES/HD132205/reduced/UVES.2010-04-02T09_28_05.650.fits.sp.ech"
    fname_py = "./Test/UVES/HD132205/reduced/2010-04-02/Reduced_middle/UVES.2010-04-02T09_28_05.650.ech"

    idl = fits.open(fname_idl)[1].data["spec"][0]
    py = fits.open(fname_py)[1].data["spec"][0][1:]

    plt.plot(py[0]/idl[0])
    plt.xlim([1000, 2000])
    plt.ylim([0.8, 1.2])
    plt.show()


if __name__ == "__main__":
    unittest.main(verbosity=2)
    # compare_idl_python()
