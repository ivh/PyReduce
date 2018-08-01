import unittest
import astropy.io.fits as fits
import numpy as np
import tempfile
import os

import combine_frames

class TestCombineMethods(unittest.TestCase):
    def setUp(self):
        self.instrument = "UVES"
        self.mode = "middle"
        n = 3
        files = [tempfile.NamedTemporaryFile(suffix=".fits", delete=False) for _ in range(n)]
        self.files = [f.name for f in files]

    def tearDown(self):
        for f in self.files:
            os.remove(f)

    def test_running_median(self):
        arr = np.array([np.arange(10), np.arange(1, 11), np.arange(2, 12)])
        size = 3

        result = combine_frames.running_median(arr, size)
        compare = np.array([np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)])

        self.assertTrue(np.array_equal(result, compare))

    def test_running_mean(self):
        arr = np.array([np.arange(10), np.arange(1, 11), np.arange(2, 12)])
        size = 3

        result = combine_frames.running_median(arr, size)
        compare = np.array([np.arange(1, 9), np.arange(2, 10), np.arange(3, 11)])

        self.assertTrue(np.array_equal(result, compare))

    def test_combine_frames(self):
        img = np.full((100, 100), 10)
        for f in self.files:
            data = img + np.random.randint(0, 20, size=img.shape)
            head = fits.Header(cards={"ESO DET OUT1 PRSCX": 0, "ESO DET OUT1 OVSCX": 5, "ESO DET OUT1 CONAD": 1, "ESO DET OUT1 RON": 0, "EXPTIME": 10, "RA": 100, "DEC": 51, "MJD-OBS": 12030})
            fits.writeto(f, data=data, header=head)

        combine, chead = combine_frames.combine_frames(self.files, "UVES", "middle", 0, window=5)

        pass