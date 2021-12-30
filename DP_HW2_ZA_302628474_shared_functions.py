import numpy as np
import cv2 as cv
import math
import os
import copy

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"  HW2 - Shared functions between all parts 
"    Point location
"     
"   Author: Ziv Aharoni
"   Date:   23/12/2020
"   Email:  ahaziv@gmail.com
"        
"   All Rights Reserved
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

class Marker:
    # a marker possesses some different defined qualities:
    # the inner radius/outer radius ratio is: 18 / 86
    # the middle radius/outer radius ratio is: 63 / 86
    def __init__(self, inp_circle_data, inp_canny_data, array_data):
        self.myIntensityThreshold = 0.5                # an intensity threshold to detect circles within the marker
        self.myCenter = inp_circle_data[:2]            # the markers' center coordinates in the picture
        self.myGloobalCenter = [0., 0., 0.]            # the markers' coordinates in the world
        self.myData = cv.threshold(array_data, 127, 255, cv.THRESH_BINARY)[1]
        self.myCannyData = inp_canny_data
        self.myNum = 1                                 # the markers' number
        self.myTolerance = 2                           # a tolerance to identify if it is truly a marker
        self.myOutRad = inp_circle_data[2]
        self.myInrRad = int(np.round(self.myOutRad * (18 / 86)))
        self.myMidRad = int(np.round(self.myOutRad * (63 / 86)))
        self.myActOutRad = 56.5                        # the actual radius in [mm]

    def find_num(self):
        center_color = self.myData[self.myOutRad, self.myOutRad]
        radius = int((self.myOutRad + self.myMidRad)/2)
        for ii in range(12):
            angle = np.pi/12 + ii * np.pi/6
            coordinates = [self.myOutRad + int(radius * np.sin(angle)), self.myOutRad + int(radius * np.cos(angle))]
            if self.myData[coordinates[0], coordinates[1]] != center_color:
                start_angle = angle
                break
            print('Unable to locate the markers starting point')
            return 0

        radius = int(3 * self.myMidRad / 4 + self.myInrRad/4)
        for ii in range(10):
            angle = start_angle + (ii+2) * np.pi/6
            coordinates = [self.myOutRad + int(radius * np.sin(angle)), self.myOutRad + int(radius * np.cos(angle))]
            if self.myData[coordinates[0], coordinates[1]] == center_color:
                self.myNum += 2 ** ii

    def is_marker(self):
        # this method tests all circles contained within the marker comply to the marker radii ratios
        center_color = self.myData[self.myOutRad, self.myOutRad]
        if center_color != self.myData[5, self.myOutRad]:
            return False
        if self.myOutRad < 22:  # for such low radii it is impossible to accurately calculate
            return False
        my_circle_array, my_circle_data = \
            find_circles(self.myCannyData, self.myInrRad - 2, self.myMidRad + 2, self.myIntensityThreshold)
        if len(my_circle_data) < 1:
            return False
        for sing_circ_dat in my_circle_data:
            if abs(sing_circ_dat[2] - self.myInrRad) < self.myTolerance:
                return True
        return False

    def set_global_center(self, coordinates):
        self.myGloobalCenter[:] = coordinates


def gradient_method(image, det_line_width):
    """
    Parameters
    ----------
    image : A greyscale image
    det_line_width : The line width

    Returns
    -------
    norm_vote_mat : matrix containing the centres for all potential circles

    """
    # obtaining the gradient
    sobel_x = cv.Sobel(image, cv.CV_64F, 1, 0, ksize=7)
    sobel_y = cv.Sobel(image, cv.CV_64F, 0, 1, ksize=7)
    img_lap = cv.Laplacian(image, cv.CV_64F)
    # filling the vote matrix
    print('Detecting circle centers\n')
    vote_mat = np.zeros(image.shape)
    for ii in range(image.shape[0]):  # y axis
        for jj in range(image.shape[1]):  # x axis
            if img_lap[ii, jj] > 40:  # fill in the line only in case of a strong gradient
                vote_mat = mark_line(vote_mat, sobel_x[ii, jj], sobel_y[ii, jj], [ii, jj], det_line_width)
    norm_vote_mat = vote_mat / np.max(np.max(vote_mat))
    return norm_vote_mat


def mark_line(vote_grid, grad_x, grad_y, coordinates, width):
    """

    Parameters
    ----------
    vote_grid : A greyscale image
    grad_x : the gradient in x
    grad_y : the gradient in y
    coordinates : the coordinates of the start point
    width : the lines' width (in pixels)

    Returns
    -------
    vote_grid : matrix containing the vote grid for all centers

    """
    if np.mod(width, 2) == 0:
        os.error('Line width must be an odd number.')
    half_width = math.floor(width / 2)
    c = -(grad_x * coordinates[0] + grad_y * coordinates[1])
    if abs(grad_x) < abs(grad_y):  # if a > b iterate over the y axis
        for kk in range(vote_grid.shape[0]):  # iterating over the y axis
            x_val = 0.5 + kk
            y_val = -(grad_x * x_val + c) / grad_y
            if 0 <= int(np.rint(y_val)) < vote_grid.shape[1]:
                vote_grid[kk, int(np.rint(y_val) - half_width):int(np.rint(y_val + half_width + 1))] += 1
    else:
        for kk in range(vote_grid.shape[1]):  # iterating over the y axis
            y_val = 0.5 + kk
            x_val = -(grad_y * y_val + c) / grad_x
            if 0 <= int(np.rint(x_val)) < vote_grid.shape[0]:
                vote_grid[int(np.rint(x_val) - half_width):int(np.rint(x_val) + half_width + 1), kk] += 1
    return vote_grid


def circle_mask(rad):
    """
    this function creates a circular array of 1 and zeros

    Parameters
    ----------
    rad : the circle radius

    Returns
    -------
    circ_imprint : The imprint of the circle

    """
    X, Y = np.mgrid[-rad:rad + 1:1, -rad:rad + 1:1]
    dist_array = np.sqrt(X ** 2 + Y ** 2)
    # circ_imprint = dist_array - rad
    circ_imprint = np.rint(dist_array)
    circ_imprint = (circ_imprint == rad).astype(np.int)
    return circ_imprint


def imprint_circle(array, imprint, x_cor, y_cor, rad):
    """
    this function imprints a circular array of 1s on a given binary image

    Parameters
    ----------
    array : the binary image
    imprint : circular imprint of 1s and 0s
    x_cor, y_cor : the column and line coordinates for the circle center
    rad : the circles' radius

    Returns
    -------
    array : The imprinted array

    """
    # cutting the imprint to fit the array
    min_x_cor = abs(min(x_cor - rad, 0))
    if x_cor + rad >= len(array[0, :]):
        max_x_cor = 2 * rad - (x_cor + rad - len(array[0, :]))
    else:
        max_x_cor = 2 * rad + 1

    min_y_cor = abs(min(y_cor - rad, 0))
    if y_cor + rad >= len(array[:, 0]):
        max_y_cor = 2 * rad - (y_cor + rad - len(array[:, 0]))
    else:
        max_y_cor = 2 * rad + 1
    cut_imrpint = imprint[min_y_cor:max_y_cor, min_x_cor:max_x_cor]
    # placing the imprint over the accumulated array
    array_x_cor = max(x_cor - rad, 0)
    array_y_cor = max(y_cor - rad, 0)
    array[array_y_cor:array_y_cor + len(cut_imrpint[:, 0]), array_x_cor:array_x_cor + len(cut_imrpint[0, :])] \
        = array[array_y_cor:array_y_cor + len(cut_imrpint[:, 0]),
          array_x_cor:array_x_cor + len(cut_imrpint[0, :])] + cut_imrpint
    return array


def find_circles(canny_input, min_rad, max_rad, intensity_threshold):
    """
    this function receives a greyscale image and extracts circles within it

    Parameters
    ----------
    canny_input : the input canny edges
    min_rad, max_rad : the minimal and maximal circle radius to search
    intensity_threshold : the intensity threshold for circle classification within the array (usually around 0.5)

    Returns
    -------
    circle_array : The vote array of all circles
    circle_data : a list containing all circle data (circle_data[0] == [y_center, x_center, radius])

    """
    edges = np.where(canny_input == 255)
    accum_array = np.zeros((canny_input.shape[0], canny_input.shape[1], max_rad))
    num_full_cells = []
    center_array = np.zeros((canny_input.shape[0], canny_input.shape[1]))

    for rad in range(min_rad, len(accum_array[0, 0, :])):
        circle = circle_mask(rad)
        num_full_cells.append(sum(sum(circle)))
        for ii in range(len(edges[0])):
            accum_array[:, :, rad] = imprint_circle(accum_array[:, :, rad], circle, edges[1][ii], edges[0][ii], rad)

    circle_data = []
    for rad in range(min_rad, len(accum_array[0, 0, :])):
        center_intensity_threshold = intensity_threshold * num_full_cells[rad - min_rad]
        while np.amax(accum_array[:, :, rad]) > center_intensity_threshold:
            max_pnt = np.unravel_index(np.argmax(accum_array[:, :, rad]),
                                       [len(accum_array[:, 0, rad]), len(accum_array[0, :, rad])])
            circle = circle_mask(rad)
            circle_array = imprint_circle(center_array, circle, max_pnt[1], max_pnt[0], rad)
            circle_data.append([max_pnt[1], max_pnt[0], rad])
            accum_array[max_pnt[0] - rad:max_pnt[0] + rad, max_pnt[1] - rad:max_pnt[1] + rad, rad] = 0
    if not circle_data:
        return [], []
    return circle_array, circle_data


def detect_markers(flagScene, flag_down_samp, down_samp_rate, min_rad, max_rad ,intensity_threshold, image):
    """
    this function creates a circular array of 1 and zeros

    Parameters
    ----------
    rad : the circle radius

    Returns
    -------
    circ_imprint : The imprint of the circle

    """
    if flag_down_samp:
        orig_img = image.copy()
        image = cv.resize(image, (0, 0), fx=down_samp_rate, fy=down_samp_rate, interpolation=cv.INTER_NEAREST)
    # scene data preparation
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if flagScene:
        # Enhance picture contrast
        gray = cv.equalizeHist(gray)
        # Reduce noise by blurring with a Gaussian filter ( kernel size = 3 )
        gray = cv.GaussianBlur(gray, (3, 3), 0)
    canny_image = cv.Canny(gray, 75, 150, apertureSize=3)
    if max_rad == 0:
        contours = cv.findContours(canny_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        max_diam = 1
        for contour in contours[0]:
            max_diam = max(
                max(max(contour[:, 0, 0]) - min(contour[:, 0, 0]), max(contour[:, 0, 1]) - min(contour[:, 0, 1])),
                max_diam)
        max_rad = int(max_diam / 2 + 3)
        # finding the circles
        print("Locating circles")
    circle_array, circle_data = find_circles(canny_image, min_rad, max_rad, intensity_threshold)

    # identifying the markers among the circles and storing them
    print("Identifying markers")
    markers = []
    for ii, data in enumerate(circle_data):
        min_x = data[0] - data[2]
        max_x = data[0] + data[2]
        min_y = data[1] - data[2]
        max_y = data[1] + data[2]
        # marker_data = np.zeros((max_y - min_y, max_x - min_x))
        marker_data = gray[min_y:max_y, min_x:max_x]
        canny_data = canny_image[min_y:max_y, min_x:max_x]
        marker = Marker(circle_data[ii], canny_data, marker_data)
        if marker.is_marker():
            marker.find_num()
            markers.append(copy.deepcopy(marker))
    return canny_image, markers


def find_proj_mat_inv(line_pairs):
    """
    this function extracts the projective rectification given two pairs of parallel lines.

    Parameters
    ----------
    line_pairs : a list of 2 line pairs, each pair contains an array of 2 lines

    Returns
    -------
    proj_mat_inv : inverse of the projective matrix

    """
    vanish_pts = np.zeros((2, 3))
    vanish_pts[0, :] = np.cross(line_pairs[0][0, :], line_pairs[0][1, :])
    vanish_pts[1, :] = np.cross(line_pairs[1][0, :], line_pairs[1][1, :])

    inf_line = np.cross(vanish_pts[0, :], vanish_pts[1, :])
    inf_line = inf_line / inf_line[2]

    projMat = np.array([[1, 0, 0],
                        [0, 1, 0],
                        [-inf_line[0], -inf_line[1], inf_line[2]]])
    proj_mat_inv = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [inf_line[0], inf_line[1], inf_line[2]]])
    return proj_mat_inv


def find_aff_mat_inv(lines, projMatInv):
    """
    this function extracts the affine rectification given two pairs of perpendicular lines.

    Parameters
    ----------
    lines : a array of 4 lines (1 in each row), where lines 0 and 2 are parallel along with lines 1 and 3
    projMatInv : the inverse of the projective matrix

    Returns
    -------
    the inverse of the affine matrix

    """
    p1 = np.matmul(projMatInv, np.cross(lines[0, :], lines[1, :]))
    p1 = p1 / p1[2]
    p2 = np.matmul(projMatInv, np.cross(lines[1, :], lines[2, :]))
    p2 = p2 / p2[2]
    p3 = np.matmul(projMatInv, np.cross(lines[2, :], lines[3, :]))
    p3 = p3 / p3[2]
    p4 = np.matmul(projMatInv, np.cross(lines[3, :], lines[0, :]))
    p4 = p4 / p4[2]

    # the orthogonally pair lines
    l1 = np.cross(p1, p2) / np.cross(p1, p2)[2]
    m1 = np.cross(p2, p3) / np.cross(p2, p3)[2]
    l2 = np.cross(p1, p3) / np.cross(p1, p3)[2]
    m2 = np.cross(p2, p4) / np.cross(p2, p4)[2]

    matA = np.array([[l1[0] * m1[0], l1[0] * m1[1] + l1[1] * m1[0]],
                     [l2[0] * m2[0], l2[0] * m2[1] + l2[1] * m2[0]]])

    sVec = np.linalg.solve(matA, np.array([[-l1[1] * m1[1]], [-l2[1] * m2[1]]]))

    sMat = np.array([[sVec[0][0], sVec[1][0]],
                     [sVec[1][0], 1.]])

    kMat = np.linalg.cholesky(sMat)
    return (np.array([[1 / kMat[0, 0], -kMat[1, 0] / (kMat[0, 0] * kMat[1, 1]), 0],
                      [0, 1 / kMat[1, 1], 0],
                      [0, 0, 1]]))