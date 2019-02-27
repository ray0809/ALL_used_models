#coding=utf-8
import cv2
import numpy as np

__author__ = "Sachin Mehta"

class LoadData:
    def __init__(self, samples, classes, normVal=1.10):
        self.samples = samples
        self.classes = classes
        self.normVal = normVal
        self.classWeights = np.ones(self.classes, dtype=np.float32)

    def compute_class_weights(self, histogram):
        '''
        Helper function to compute the class weights
        :param histogram: distribution of class samples
        :return: None, but updates the classWeights variable
        '''
        normHist = histogram / np.sum(histogram)
        for i in range(self.classes):
            self.classWeights[i] = 1 / (np.log(self.normVal + normHist[i]))

    def readFiles(self):
        global_hist = np.zeros(self.classes, dtype=np.float32)
        for imgname, maskname in self.samples:
            mask = cv2.imread(maskname, 0)
            hist = np.histogram(mask, self.classes)
            global_hist += hist[0]
        self.compute_class_weights(global_hist)