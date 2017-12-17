import numpy as np
from scipy.signal import fftconvolve

# Accepts a matrix and downsamples it according to the parity given.  We need this for the DWT.
# parity = 0 => discard the odd-index rows and columns


def _downsample(matrix, offset):
    return matrix[0 + offset::2, 0 + offset::2]

# Accepts a matrix and returns an upsampled version of the matrix.
# #This is needed for the inverse DWT.
# sx and sy are the desired final length
# parity determines the choice of above/below in adding zeros (or left/right for columns)
# parity = 1 => left and upper padding


def _upsample(matrix, size, offset):
    temp = np.zeros(size)
    temp[0 + offset::2, 0 + offset::2] = matrix
    return temp

# Structure to store wavelet coefficients


class WaveletCoeffs:

    # Given the image size, calculates the sizes of the coefficient matrices
    # at levels 1 to N+1
    def __getsizes(self, itr):  # parity=0 => start from zero, ignore alternate indices
        if itr == self.levels + 1:
            return
        ver, horiz = self.sizes[-1]

        # get size after convolution with the masks
        ver, horiz = ver + self.masklen - 1, horiz + self.masklen - 1
        ver1, horiz1 = ver // 2, horiz // 2
        # for even-length masks, we need to discard the even index rows/columns
        parity = (self.masklen + 1) % 2
        if parity == 0 and ver % 2 == 1:
            ver1 += 1  # we need an additional row/column if the total is odd and we start from zero
        if parity == 0 and horiz % 2 == 1:
            horiz1 += 1
        self.sizes.append((ver1, horiz1))
        self.__getsizes(itr + 1)

    def __init__(self, masks, levels, size):
        self.masks = np.array(masks, dtype=np.float64)
        self.nummasks, self.masklen = self.masks.shape
        # number of levels to traverse down. image = level 0, followed by -1 to
        # -levels of coefficients
        self.levels = levels
        # the size of the coefficient arrays at each stage
        self.sizes = [size, ]
        self.__getsizes(1)
        self.coeffs = np.empty(levels + 1, dtype=object)
        for i in range(levels + 1):
            if i == 0:
                self.coeffs[i] = np.empty((1, 1) + size)
            else:
                self.coeffs[i] = np.empty(
                    (self.nummasks, self.nummasks) + self.sizes[i])

# overload the '+' operator for wavelet coefficients
# WARNING: does not check if the masks for the coefficient objects are the same.
# If they are different, the masks for the left operand are chosen.
    def __add__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs + other.coeffs
        return result

    # overload the '-' operator for wavelet coefficients
    # Same warning as for '+' operator
    def __sub__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs - other.coeffs
        return result

    # overload the '*' operator for wavelet coefficients
    # Same warning as for '+' operator
    def __mul__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs * other
        return result

    def __truediv__(self, other):
        result = WaveletCoeffs(self.masks, self.levels, self.sizes[0])
        result.coeffs = self.coeffs / other
        return result

    # overload element access
    def __getitem__(self, key):
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, list):
            return self.coeffs[key]
        elif isinstance(key, tuple):
            if len(key) == 1:
                return self.coeffs[key]
            elif len(key) > 1:
                return self.coeffs[key[0]][key[1:]]
        else:
            raise IndexError(
                'Index must be an integer, a slice, a list of integers or a tuple of indices')

    def __setitem__(self, key, value):
        if isinstance(key, int) or isinstance(key, slice) or isinstance(key, list):
            self.coeffs[key] = value
        elif isinstance(key, tuple):
            if len(key) == 1:
                self.coeffs[key] = value
            elif len(key) > 1:
                self.coeffs[key[0]][key[1:]] = value
        else:
            raise IndexError(
                'Index must be an integer, a slice, a list of integers or a tuple of indices')

    # Inverse Discrete Wavelet Transform for a 2D matrix
    def invdwt2(self):
        temp = self[self.levels, 0, 0]
        for cur_level in range(self.levels, 0, -1):
            nummasks = self.nummasks
            masks = self.masks
            masklen = self.masklen
            offset = (masklen + 1) % 2

            # calculate the right size for upsampling
            final_x, final_y = self.sizes[cur_level]
            prev_x, prev_y = self.sizes[cur_level - 1]
            final_x, final_y = 2 * final_x, 2 * final_y

            # the convolved matrix before downsampling had odd number of rows (X+masklen-1)
            # and we discarded the even numbered ones
            if prev_x % 2 == 0 and masklen % 2 == 0:
                final_x += 1
            # the matrix had an odd number of rows and we discarded the odd
            # numbered ones
            elif prev_x % 2 == 1 and masklen % 2 == 1:
                final_x -= 1
            if prev_y % 2 == 0 and masklen % 2 == 0:  # similar
                final_y += 1
            elif prev_y % 2 == 1 and masklen % 2 == 1:  # similar
                final_y -= 1
            for mask1 in range(nummasks):
                for mask2 in range(nummasks):
                    if mask1 == mask2 == 0:
                        temp = 2 * fftconvolve(
                            masks[mask1, :, None] * masks[mask2],
                            _upsample(temp, (final_x, final_y), offset))[
                                masklen - 1:-(masklen - 1), (masklen - 1):-(masklen - 1)]
                    else:
                        temp += 2 * fftconvolve(
                            masks[mask1, :, None] * masks[mask2],
                            _upsample(self[cur_level, mask1, mask2], (final_x, final_y), offset))[
                                masklen - 1:-(masklen - 1), (masklen - 1):-(masklen - 1)]
        return temp

# Discrete Wavelet Transform (DWT) of a 2D matrix


def dwt2(img, masks, levels):
    img = np.array(img)
    masks = np.array(masks)
    nummasks, masklen = masks.shape  # number of masks and length of each mask
    offset = (masklen + 1) % 2
    imgdwt = WaveletCoeffs(masks, levels, img.shape)
    imgdwt[0, 0, 0] = img
    for cur_level in range(1, levels + 1):
        for mask1 in range(nummasks):
            for mask2 in range(nummasks):
                imgdwt[cur_level, mask1, mask2] = _downsample(
                    2 * fftconvolve(masks[mask1, -1::-1, None] * masks[mask2, -1::-1],
                                    imgdwt[cur_level - 1, 0, 0]), offset)

    return imgdwt
