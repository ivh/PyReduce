import numpy as np
import pytest

from pyreduce.clipnflip import clipnflip

pytestmark = pytest.mark.unit

# clipnflip(image, header, xrange=None, yrange=None, orientation=None)


@pytest.fixture
def image():
    image = np.arange(0, 200).reshape((20, 10))
    return image


@pytest.fixture
def header(image):
    # Default header is set up to change nothing
    # Note that image indices are y and x (and not x and y)
    header = {}
    header["e_amp"] = 1
    header["e_xlo"], header["e_xhi"] = 0, image.shape[1]
    header["e_ylo"], header["e_yhi"] = 0, image.shape[0]
    header["e_orient"] = 0
    header["e_transpose"] = False
    return header


def test_nochange(image, header):
    # This should change nothing
    flipped = clipnflip(image, header)

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == image.shape[0]
    assert flipped.shape[1] == image.shape[1]
    assert np.all(flipped == image)


def test_only_rotate(image, header):
    for orient in [0, 1, 2, 3]:
        flipped = clipnflip(image, header, orientation=orient)

        compare = np.rot90(image, -1 * orient)

        assert isinstance(flipped, np.ndarray)
        assert flipped.shape[0] == compare.shape[0]
        assert flipped.shape[1] == compare.shape[1]
        assert np.all(flipped == compare)


def test_only_clip(image, header):
    # clip x direction
    flipped = clipnflip(image, header, xrange=(5, 8))

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == 20
    assert flipped.shape[1] == 3
    assert np.all(flipped == image[:, 5:8])

    # clip y direction
    flipped = clipnflip(image, header, yrange=(10, 15))

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == 5
    assert flipped.shape[1] == 10
    assert np.all(flipped == image[10:15, :])


def test_bad_clipping_range(image, header):
    # Case 1: x completely out of range
    with pytest.raises(IndexError):
        clipnflip(image, header, xrange=(100, 150))

    # Case 2: x upper limit out of range
    with pytest.raises(IndexError):
        clipnflip(image, header, xrange=(5, 150))

    # Case 3: x limits inverse
    with pytest.raises(IndexError):
        clipnflip(image, header, xrange=(10, 0))

    # Case 4: x limits only one column
    with pytest.raises(IndexError):
        clipnflip(image, header, xrange=(1, 1))

    # Case 1: y completely out of range
    with pytest.raises(IndexError):
        clipnflip(image, header, yrange=(100, 150))

    # Case 2: y upper limit out of range
    with pytest.raises(IndexError):
        clipnflip(image, header, yrange=(5, 150))

    # Case 3: y limits inverse
    with pytest.raises(IndexError):
        clipnflip(image, header, yrange=(10, 0))

    # Case 4: y limits only one column
    with pytest.raises(IndexError):
        clipnflip(image, header, yrange=(1, 1))


def test_multidimensional(image, header):
    image = image[None, None, ...]
    flipped = clipnflip(image, header)

    assert isinstance(flipped, np.ndarray)
    assert flipped.shape[0] == image.shape[-2]
    assert flipped.shape[1] == image.shape[-1]
    assert np.all(flipped == image[0, 0])


def test_orientation_with_transpose(image):
    """Test orientations 4-7 which involve transpose before rotation."""
    # Use header without e_transpose so the default (orient >= 4) kicks in
    header = {
        "e_amp": 1,
        "e_xlo": 0,
        "e_xhi": image.shape[1],
        "e_ylo": 0,
        "e_yhi": image.shape[0],
        "e_orient": 0,
    }
    for orient in [4, 5, 6, 7]:
        flipped = clipnflip(image, header, orientation=orient)

        # For orient >= 4, transpose is applied first, then rotation
        compare = np.transpose(image)
        compare = np.rot90(compare, -1 * orient)

        assert isinstance(flipped, np.ndarray)
        assert flipped.shape == compare.shape
        assert np.all(flipped == compare)


def test_orientation_from_header(image, header):
    """Test that orientation is read from header when not passed explicitly."""
    for orient in [0, 1, 2, 3]:
        header["e_orient"] = orient
        header["e_transpose"] = False
        flipped = clipnflip(image, header)

        compare = np.rot90(image, -1 * orient)

        assert flipped.shape == compare.shape
        assert np.all(flipped == compare)


def test_transpose_from_header(image, header):
    """Test that transpose is read from header."""
    header["e_orient"] = 0
    header["e_transpose"] = True
    flipped = clipnflip(image, header)

    compare = np.transpose(image)
    assert flipped.shape == compare.shape
    assert np.all(flipped == compare)


def test_header_transpose_overrides_default(image):
    """Test that e_transpose in header overrides the default (orient >= 4) logic."""
    header = {
        "e_amp": 1,
        "e_xlo": 0,
        "e_xhi": image.shape[1],
        "e_ylo": 0,
        "e_yhi": image.shape[0],
        "e_orient": 5,
        "e_transpose": False,  # Override default (would be True for orient=5)
    }
    flipped = clipnflip(image, header)

    # With transpose=False, only rotation is applied
    compare = np.rot90(image, -5)
    assert flipped.shape == compare.shape
    assert np.all(flipped == compare)


def test_all_eight_orientations():
    """
    Test all 8 orientation values produce the expected transformations.

    Orientation encoding (IDL convention):
    - Bits 0-1: number of 90° clockwise rotations (0-3)
    - Bit 2: transpose before rotation (orientations 4-7)

    Using a non-square asymmetric image to verify transformations.
    """
    # Create asymmetric image where we can verify exact transformation
    # 3 rows x 4 cols, unique values at corners
    image = np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]])
    header = {
        "e_amp": 1,
        "e_xlo": 0,
        "e_xhi": 4,
        "e_ylo": 0,
        "e_yhi": 3,
    }

    # Expected results for each orientation
    # orient 0: no change
    expected_0 = image.copy()

    # orient 1: rotate 90° CW (np.rot90 with k=-1)
    expected_1 = np.rot90(image, -1)

    # orient 2: rotate 180°
    expected_2 = np.rot90(image, -2)

    # orient 3: rotate 270° CW (= 90° CCW)
    expected_3 = np.rot90(image, -3)

    # orient 4: transpose only
    expected_4 = np.transpose(image)

    # orient 5: transpose then rotate 90° CW
    expected_5 = np.rot90(np.transpose(image), -1)

    # orient 6: transpose then rotate 180°
    expected_6 = np.rot90(np.transpose(image), -2)

    # orient 7: transpose then rotate 270° CW
    expected_7 = np.rot90(np.transpose(image), -3)

    expectations = [
        expected_0,
        expected_1,
        expected_2,
        expected_3,
        expected_4,
        expected_5,
        expected_6,
        expected_7,
    ]

    for orient, expected in enumerate(expectations):
        result = clipnflip(image, header, orientation=orient)
        assert result.shape == expected.shape, (
            f"orientation={orient}: shape {result.shape} != expected {expected.shape}"
        )
        assert np.all(result == expected), (
            f"orientation={orient}: values don't match\n"
            f"got:\n{result}\nexpected:\n{expected}"
        )


def test_orientation_corners():
    """
    Verify corner positions after each orientation transform.
    This provides an intuitive check that orientations work as expected.
    """
    # Image with labeled corners: TL=1, TR=2, BL=3, BR=4
    image = np.array([[1, 0, 2], [0, 0, 0], [3, 0, 4]])
    header = {
        "e_amp": 1,
        "e_xlo": 0,
        "e_xhi": 3,
        "e_ylo": 0,
        "e_yhi": 3,
    }

    def get_corners(img):
        """Return (top-left, top-right, bottom-left, bottom-right)."""
        return (img[0, 0], img[0, -1], img[-1, 0], img[-1, -1])

    # Original: TL=1, TR=2, BL=3, BR=4
    assert get_corners(image) == (1, 2, 3, 4)

    # orient 0: no change
    r = clipnflip(image, header, orientation=0)
    assert get_corners(r) == (1, 2, 3, 4)

    # orient 1: 90° CW - left side becomes top
    r = clipnflip(image, header, orientation=1)
    assert get_corners(r) == (3, 1, 4, 2)

    # orient 2: 180° - flip both axes
    r = clipnflip(image, header, orientation=2)
    assert get_corners(r) == (4, 3, 2, 1)

    # orient 3: 270° CW (90° CCW) - right side becomes top
    r = clipnflip(image, header, orientation=3)
    assert get_corners(r) == (2, 4, 1, 3)

    # orient 4: transpose only - swap rows/cols
    r = clipnflip(image, header, orientation=4)
    assert get_corners(r) == (1, 3, 2, 4)

    # orient 5: transpose + 90° CW
    r = clipnflip(image, header, orientation=5)
    assert get_corners(r) == (2, 1, 4, 3)

    # orient 6: transpose + 180°
    r = clipnflip(image, header, orientation=6)
    assert get_corners(r) == (4, 2, 3, 1)

    # orient 7: transpose + 270° CW
    r = clipnflip(image, header, orientation=7)
    assert get_corners(r) == (3, 4, 1, 2)
