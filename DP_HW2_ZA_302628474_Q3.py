import os
import sys
import matplotlib.pyplot as plt
sys.path.insert(1, os.path.join(os.getcwd(), '..\\'))
from DP_HW2_ZA_302628474_shared_functions import *

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
"   HW2 - Question 3 
"   Point location
"     
"   Author: Ziv Aharoni
"   Date:   23/12/2020
"   Email:  ahaziv@gmail.com
"        
"   All Rights Reserved
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""


class Cursor:
    # a cursor possesses some different defined qualities:
    def __init__(self):
        self.myActPts = np.array([[0, -43], [0, 87], [34.5, 16], [34.5, 16]])
        self.myActRad = 29.5
        self.myActTip = [0, 136]
        self.myActCtr = [0, 16]
        self.myDigCtr = 0
        self.myDigPts = []
        self.myDigRad = 0
        self.myDigTip = [0, 0]
        self.myDigActRatio = 0

    def locate_dig_pts(self, canny_input, center_marker, intensity_threshold):
        detection_thresh = 1
        center = center_marker.myCenter
        self.myDigActRatio = center_marker.myOutRad / self.myActRad  # used to transfer from digital to actual
        dig_dist = []
        for ii in range(len(self.myActPts[:, 0])):
            dig_dist.append(np.linalg.norm(self.myActPts[ii, :]) * self.myDigActRatio)
        # locating the circles in the vicinity of the marker
        circle_array, circle_data = find_circles(canny_input, center_marker.myMidRad - 2, center_marker.myMidRad + 3,
                                                 intensity_threshold)

        dig_pts = np.zeros((len(self.myActPts[:, 0]), 2))
        for ii, dist in enumerate(dig_dist):
            for jj, circle in enumerate(circle_data):
                if (dist - np.linalg.norm(np.array([circle[0] - center[0], circle[1] - center[1]]))) < detection_thresh:
                    dig_pts[ii, :] = circle[0], circle[1]
                    self.myDigPts.append(circle_data.pop(jj))
                    break
        if len(self.myDigPts) < 4:
            print('could not detect cursor')
            return 0
        # if the two side circles are flipped switch between them
        if (dig_pts[1, 0] - dig_pts[0, 0]) / (dig_pts[1, 1] - dig_pts[0, 1]) < \
                (dig_pts[2, 0] - dig_pts[0, 0]) / (dig_pts[2, 1] - dig_pts[0, 1]):
            temp = dig_pts[3, :]
            dig_pts[3, :] = dig_pts[2, :]
            dig_pts[2, :] = temp
            temp = self.myDigPts[3]
            self.myDigPts[3] = self.myDigPts[2]
            self.myDigPts[2] = temp

    def find_dig_tip(self):
        if self.myDigActRatio == 0:
            raise RuntimeError('Cursor parameters must be found before find_dig_tip is attempted')
        direction = np.asarray(self.myDigPts[1][:2]) - np.asarray(self.myDigPts[0][:2])
        direction = np.true_divide(direction, np.linalg.norm(direction))
        magnitude = self.myActTip - np.asarray(self.myActPts[0])
        magnitude = np.linalg.norm(magnitude)
        self.myDigTip = np.asarray(self.myDigPts[0][:2]) + direction * magnitude * self.myDigActRatio * 2
        return self.myDigTip

    def find_dig_par_pts(self):
        if self.myDigActRatio == 0:
            raise RuntimeError('Cursor parameters must be found before find_dig_tip is attempted')
        direction = np.asarray(self.myDigPts[1][:2]) - np.asarray(self.myDigPts[0][:2])
        direction = np.true_divide(direction, np.linalg.norm(direction))
        magnitude = self.myActCtr[1]
        self.myDigCtr = np.asarray(self.myDigPts[0][:2]) + direction * magnitude * self.myDigActRatio * 2
        magnitude = self.myActPts[3, 0]
        pnt1 = np.asarray(self.myDigCtr) - direction * magnitude * self.myDigActRatio * 2
        pnt2 = np.asarray(self.myDigCtr) + direction * magnitude * self.myDigActRatio * 2
        return pnt1, pnt2


def transformation_matrix(points):
    """
    the four points:
    0.______________.1
     |              |
     |              |
     |              |
    2.______________.3
    # By use of these points this function calculate the transformation matrix to the camera plane

    Parameters
    ----------
    The transformation matrix

    Returns
    -------
    The warped image

    """
    vec_lines = np.zeros((4, 3))
    vec_lines[0, :] = np.cross(points[0, :], points[1, :])
    vec_lines[0, :] /= vec_lines[0, 2]
    vec_lines[1, :] = np.cross(points[2, :], points[3, :])
    vec_lines[1, :] /= vec_lines[1, 2]
    vec_lines[2, :] = np.cross(points[0, :], points[2, :])
    vec_lines[2, :] /= vec_lines[2, 2]
    vec_lines[3, :] = np.cross(points[1, :], points[3, :])
    vec_lines[3, :] /= vec_lines[3, 2]

    parLines = [np.ones((2, 3)), np.ones((2, 3))]
    parLines[0][0, :] = vec_lines[0, :]
    parLines[0][1, :] = vec_lines[1, :]
    parLines[1][0, :] = vec_lines[2, :]
    parLines[1][1, :] = vec_lines[3, :]
    projMatInv = find_proj_mat_inv(parLines)

    tempVecLines = vec_lines.copy()
    tempVecLines[1, :] = vec_lines[2, :]
    tempVecLines[2, :] = vec_lines[1, :]
    affineMatInv = find_aff_mat_inv(tempVecLines, projMatInv)

    return np.matmul(projMatInv, affineMatInv)


# -------------------------------------------------- main ------------------------------------------------------------ #
if __name__ == "__main__":
    flagScene = True  # This flag states the mode of execution - training the kMeans or testing the segmentation
    flag_down_samp = False
    down_samp_rate = 0.25
    min_rad = 5
    max_rad = 37
    int_thresh = 0.45  # an intensity threshold for the circle detection
    marker_num = 6
    markers2_4_ver_dist = 240

    # import pictures
    data_path = os.path.join(os.getcwd(), 'Scenes data\\')
    file_name = 'Scene6'
    if os.path.isfile(file_name + '.jpeg'):
        img1 = cv.imread(file_name + '.jpeg')
    else:
        img1 = cv.imread(data_path + file_name + '.jpeg')

    canny_image, markers = detect_markers(flagScene, flag_down_samp, down_samp_rate, min_rad, max_rad, int_thresh, img1)
    # sort markers
    sorted_markers = [[]] * marker_num
    for marker in markers:
        if marker.myNum > marker_num:
            raise RuntimeError('Markers have too high numbers, please adjust parameter/check artifacts')
        if not sorted_markers[marker.myNum - 1]:
            sorted_markers[marker.myNum - 1] = marker
    # test if all slots are full
    if not all(sorted_markers):
        print('Could not locate all markers in picture.')
    temp = abs((sorted_markers[3].myCenter[0] - sorted_markers[1].myCenter[0]))
    dig_atc_ratio = abs((sorted_markers[3].myCenter[0] - sorted_markers[1].myCenter[0])) / markers2_4_ver_dist
    marker_act_coor = np.zeros((len(sorted_markers), 3))
    for ii, marker in enumerate(sorted_markers[2:]):
        marker_act_coor[ii + 2, 0] = (marker.myCenter[0] - sorted_markers[1].myCenter[0]) * dig_atc_ratio
        marker_act_coor[ii + 2, 1] = (marker.myCenter[1] - sorted_markers[1].myCenter[1]) * dig_atc_ratio

    cursor1 = Cursor()
    cursor1.locate_dig_pts(canny_image, sorted_markers[0], int_thresh)
    cursor_tip = cursor1.find_dig_tip()

    # find the transformation matrices
    img_curs_pnt1, img_curs_pnt2 = cursor1.find_dig_par_pts()
    curs_pts = np.array([np.concatenate([img_curs_pnt1, np.array([1])], axis=0),
                         np.concatenate([cursor1.myDigPts[2][:2], np.array([1])], axis=0),
                         np.concatenate([cursor1.myDigPts[3][:2], np.array([1])], axis=0),
                         np.concatenate([img_curs_pnt2, np.array([1])], axis=0)])
    curs_trans_mat = transformation_matrix(curs_pts)
    wall_pts = np.array([np.concatenate([sorted_markers[1].myCenter, np.array([1])], axis=0),
                         np.concatenate([sorted_markers[3].myCenter, np.array([1])], axis=0),
                         np.concatenate([sorted_markers[4].myCenter, np.array([1])], axis=0),
                         np.concatenate([sorted_markers[5].myCenter, np.array([1])], axis=0)])
    wall_trans_mat = transformation_matrix(wall_pts)

    # find the cursor tip coordinates
    wall_coordinates = np.array([1, 1, 1000])
    wall_coordinates[:2] = cursor_tip - np.asarray(img1.shape[:2])
    dist_ratio = cursor1.myDigActRatio / (markers[1].myActOutRad / 8)
    real_coordinates = wall_coordinates * dist_ratio

    # plotting
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.axes.xaxis.set_ticks([])
    ax.axes.yaxis.set_ticks([])
    ax.set_title("Measurement platform", fontsize=14)
    ax.imshow(cv.cvtColor(img1, cv.COLOR_BGR2RGB), cmap='gray', origin='upper')
    for marker in sorted_markers:
        if marker:
            circle = plt.Circle((marker.myCenter[0], marker.myCenter[1]), marker.myOutRad, color='r', fill=False)
            ax.add_artist(circle)
            ax.text(marker.myCenter[0], marker.myCenter[1], str(marker.myNum), fontsize=12, color='r')

    for ii, circle in enumerate(cursor1.myDigPts):
        art_circ = plt.Circle((circle[0], circle[1]), circle[2], color='b', fill=False)
        ax.add_artist(art_circ)
        ax.text(circle[0], circle[1], str(ii), fontsize=12, color='b')

    art_circ = plt.Circle((cursor_tip[0], cursor_tip[1]), 2, color='g', fill=False)
    ax.add_artist(art_circ)
    text = (" [%.f, %.f, %.f]" % (real_coordinates[0], real_coordinates[1], real_coordinates[2]))
    ax.text(cursor_tip[0], cursor_tip[1], text, fontsize=9, color='g')

    plt.show()

