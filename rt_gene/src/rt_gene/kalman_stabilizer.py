"""
MIT License

Copyright (c) 2017 Yin Guobing

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import numpy as np
import cv2

"""
Code from https://github.com/yinguobing/head-pose-estimation
Using Kalman Filter as a point stabilizer to stabilize a 2D point.
"""

class Stabilizer(object):
    """Using Kalman filter as a point stabilizer."""

    def __init__(self,
                 state_num=4,
                 measure_num=2,
                 cov_process=0.0001,
                 cov_measure=0.1):
        """Initialization"""
        assert state_num == 4 or state_num == 2, "Only scalar and point supported, check state_num please."

        self.state_num = state_num
        self.measure_num = measure_num

        self.filter = cv2.KalmanFilter(state_num, measure_num, 0)

        # Make sure these are float32!
        self.state = np.zeros((state_num, 1), dtype=np.float32)
        self.measurement = np.zeros((measure_num, 1), dtype=np.float32)
        self.prediction = np.zeros((state_num, 1), dtype=np.float32)

        if self.measure_num == 1:
            self.filter.transitionMatrix = np.array([[1, 1],
                                                     [0, 1]], dtype=np.float32)
            self.filter.measurementMatrix = np.array([[1, 1]], dtype=np.float32)
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1]], dtype=np.float32) * cov_measure

        if self.measure_num == 2:
            self.filter.transitionMatrix = np.array([[1, 0, 1, 0],
                                                     [0, 1, 0, 1],
                                                     [0, 0, 1, 0],
                                                     [0, 0, 0, 1]], dtype=np.float32)
            self.filter.measurementMatrix = np.array([[1, 0, 0, 0],
                                                      [0, 1, 0, 0]], dtype=np.float32)
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], dtype=np.float32) * cov_measure

    def update(self, measurement):
        """Update the filter"""
        self.prediction = self.filter.predict()

        # Convert to float32 measurement
        if self.measure_num == 1:
            self.measurement = np.array([[float(measurement[0])]], dtype=np.float32)
        else:
            self.measurement = np.array([[float(measurement[0])],
                                         [float(measurement[1])]], dtype=np.float32)

        self.filter.correct(self.measurement)
        self.state = self.filter.statePost

    def set_q_r(self, cov_process=0.1, cov_measure=0.001):
        """Set process and measurement noise covariances"""
        if self.measure_num == 1:
            self.filter.processNoiseCov = np.array([[1, 0],
                                                    [0, 1]], dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1]], dtype=np.float32) * cov_measure
        else:
            self.filter.processNoiseCov = np.array([[1, 0, 0, 0],
                                                    [0, 1, 0, 0],
                                                    [0, 0, 1, 0],
                                                    [0, 0, 0, 1]], dtype=np.float32) * cov_process
            self.filter.measurementNoiseCov = np.array([[1, 0],
                                                        [0, 1]], dtype=np.float32) * cov_measure