import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import sys
import scipy as sp
import statsmodels.tsa.arima.model as ar

import math
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from numpy.random import rand, multivariate_normal

np.set_printoptions(precision=2, suppress=True, threshold=10000, edgeitems=10, linewidth=1000)

class Qdetector(object):

    def __init__(self, threshold):
        self.threshold = threshold

        self.detectPoint = None
        self.changePoint = None
        self.Rkn_max = None
        self.detectionFlag = False

    def detect(self, data):
        # self.data = np.diff(data)
        self.data = data
        try:
            dim, length = self.data.shape
        except ValueError:
            self.data = np.expand_dims(self.data, axis=0)
            dim, length = self.data.shape

        self.dimensionality = dim
        self.numSteps = length

        self.DMat = np.zeros((self.dimensionality, self.numSteps, self.numSteps))
        self.Rn_of_Zi = np.zeros((self.dimensionality, self.numSteps))
        self.Rn_bar_nk = np.zeros((self.dimensionality, self.numSteps))
        self.Rkn = np.zeros((1, self.numSteps))

        Z1toN = self.data.copy()

        # # 将Series转换为二维数组
        # array = Z1toN.reshape(-1, 1)
        #
        # # 创建MinMaxScaler对象进行归一化
        # scaler = MinMaxScaler()
        #
        # # 进行归一化
        # normalized_array = scaler.fit_transform(array)
        #
        # # 将归一化后的数组转换回Series
        # Z1toN = normalized_array.flatten().reshape(self.dimensionality,  self.numSteps)

        if dim == 1:
            for idx in range(length):
                Z1toN[0][idx] = Z1toN[0][idx] + math.log(idx+1)/100000000

        n = self.numSteps
        for i in np.arange(n):
            for j in np.arange(n):
                if i !=j:
                    self.DMat[:, i, j] = (Z1toN[:, i] - Z1toN[:, j])/np.linalg.norm(Z1toN[:, i] - Z1toN[:, j])

        self.Rn_of_Zi = np.sum(self.DMat, axis=2)

        for k in np.arange(n):
            self.Rn_bar_nk[:, k] = (1.0/(k+1))*np.sum(self.Rn_of_Zi[:, 0:k+1], axis=1)

        Cov_Mat_Est = np.zeros((self.dimensionality, self.dimensionality, n));
        for k in np.arange(n):#assuming the last data point cannot be detected as a change point, since covariance matrix will become a zero matrix.
            Cov_Mat_Est[:, :, k] = (n-(k+1))/((n-1.0)*n*(k+1))*np.matmul(self.Rn_of_Zi, self.Rn_of_Zi.transpose())
            # print(Cov_Mat_Est[:,:,k])
            inv_Cov = np.linalg.inv(Cov_Mat_Est[:, :, k]+0.000001*np.eye(self.dimensionality, dtype=np.float64))
            self.Rkn[0, k] = np.matmul(np.matmul(self.Rn_bar_nk[:, k].transpose(), inv_Cov), self.Rn_bar_nk[:, k])

        RknVar = np.var(self.Rkn, axis=1)
        RknMean = np.mean(self.Rkn, axis=1)
        # todo
        self.changePoint = np.argmax(self.Rkn)

        print(RknVar/RknMean)
        anomalies = pd.Series(dtype=float)

        if RknVar/RknMean >= self.threshold:
            self.detectionFlag = True
            print("Change point detected at sample # %d." % self.changePoint)
            anomalies = anomalies.append(pd.Series(data=[data[self.changePoint]], index=[self.changePoint]))
        # anomalies = anomalies.append(pd.Series(data=[data[self.changePoint]], index=[self.changePoint]))

        return anomalies
        # if self.Rkn[0, self.changePoint] >= self.threshold:
        #     self.detectionFlag = True
        #     print("Change point detected at sample # %d.", self.changePoint)

def main():
    #Choose one of the following state vectors as needed.
    # stateVector = np.asarray([1,1,0,1,0,0,1,1,1,0,1,1,1,1,0,0])
    stateVector = np.asarray([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1])
    stateVector = np.asarray([0,0,0,0,0,1])
    stateVector = np.asarray([0,0,1,1,0,0])
    stateVector = np.asarray([0,1])
    numOfEvents = len(stateVector)
    # stateVector = np.random.randint(2, size=numOfEvents)
    sojournTimeVector = np.random.randint(1, high=300, size=numOfEvents)
    sojournTimeVector[-1] = 20
    groundTruth = np.repeat(stateVector, sojournTimeVector)
    numberOfSamples = np.sum(sojournTimeVector)

    sigma = 0.8
    np.random.seed()
    samples1 = sigma * np.random.randn(groundTruth.size) + groundTruth
    # samples2 = sigma * np.random.randn(groundTruth.size) + np.zeros_like(groundTruth)
    # samples3 = sigma * np.random.randn(groundTruth.size) + np.zeros_like(groundTruth)
    # samples4 = sigma * np.random.randn(groundTruth.size) + np.zeros_like(groundTruth)
    samples2 = sigma * np.random.randn(groundTruth.size) + groundTruth
    samples3 = sigma * np.random.randn(groundTruth.size) + groundTruth
    samples4 = sigma * np.random.randn(groundTruth.size) + groundTruth

    samples = np.vstack((samples1, samples2, samples3, samples4))
    # samples = samples1
    threshold = numberOfSamples/3.0
    detector = Qdetector(30, samples)
    detector.detect()

    f, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
    ax1.plot(groundTruth.T, '-o')
    ax1.plot(detector.changePoint, groundTruth[detector.changePoint], '-ro')
    ax2.plot(samples.T, '-o')
    ax3.plot(detector.Rkn.T, '-o')
    ax3.plot(detector.changePoint, detector.Rkn[0, detector.changePoint], '-ro')
    ax1.set(xlabel='sequence number', ylabel='State value',)
    ax2.set(xlabel='sequence number', ylabel='Sample value',)
    ax3.set(xlabel='sequence number', ylabel='Test statistic Rkn')
    ax1.grid()
    ax2.grid()
    ax3.grid()
    plt.show()

class change_finder(object):
    # Costructor

    def __init__(self, term=70, window=5, order=(1, 1, 0)):
        # @brief Quantity for learning
        self.term = term
        # @brief Smoothing width
        self.window = window
        # @brief Order for ARIMA model
        self.order = order
        print("term:", term, "window:", window, "order:", order)

    # Main Function
    # @param[in] X Data Set
    # @return score vector
    def main(self, X):
        req_length = self.term * 2 + self.window + np.round(self.window / 2) - 2
        if len(X) < req_length:
            sys.exit("ERROR! Data length is not enough.")

        print("Scoring start.")
        # X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        score = self.outlier(X)
        score = self.changepoint(score)

        space = np.zeros(len(X) - len(score))
        score = np.r_[space, score]
        print("Done.")

        return score

    # Calculate Outlier Score from Data
    # @param[in] X Data Set
    # @return Outlier-score (M-term)-vector
    def outlier(self, X):
        count = len(X) - self.term - 1

        # train
        trains = [X[t:(t + self.term)] for t in range(count)]
        target = [X[t + self.term + 1] for t in range(count)]
        fit = [ar.ARIMA(trains[t], order=self.order).fit() for t in range(count)]

        # predict
        resid = [fit[0].forecast(1)[0] - target[t] for t in range(count)]
        pred = [fit[t].predict() for t in range(count)]
        m = np.mean(pred, axis=1)
        s = np.std(pred, axis=1)

        # logloss
        score = -sp.stats.norm.logpdf(resid, m, s)

        # smoothing
        score = self.smoothing(score, self.window)

        return score

    # Calculate ChangepointScore from OutlierScore
    # @param[in] X Data Set(Outlier Score)
    # @return Outlier-score (M-term)-vector
    def changepoint(self, X):
        count = len(X) - self.term - 1

        trains = [X[t:(t + self.term)] for t in range(count)]
        target = [X[t + self.term + 1] for t in range(count)]
        m = np.mean(trains, axis=1)
        s = np.std(trains, axis=1)

        score = -sp.stats.norm.logpdf(target, m, s)
        score = self.smoothing(score, int(np.round(self.window / 2)))

        return score

    # Calculate ChangepointScore from OutlierScore
    # @param[in] X Data set
    # @param[in] w Window size
    # @return Smoothing-score
    def smoothing(self, X, w):
        return np.convolve(X, np.ones(w) / w, 'valid')


def sample():
    from numpy.random import rand, multivariate_normal
    data_a = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)
    data_b = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)
    data_c = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)

    data_d = multivariate_normal(rand(1) * 20 - 10, np.eye(1) * (rand()), 250)
    X = np.r_[data_a, data_b, data_c, data_d][:, 0]
    c_cf = change_finder(term=70, window=7, order=(2, 2, 0))
    result = c_cf.main(X)
    return result

if __name__ == '__main__':
    ## generating sample data
    data_a = multivariate_normal([10.0], np.eye(1) * 0.1, 400)
    data_b = multivariate_normal(rand(1) * 100, np.eye(1), 100)
    X = np.r_[data_a, data_b][:, 0]

    ## scoring
    c_cf = change_finder(term=10, window=50, order=(1, 1, 0))
    result = c_cf.main(X)

    ## plot
    fig = plt.figure()
    axL = fig.add_subplot(111)
    line1, = axL.plot(X, "b-", alpha=.7)
    plt.ylabel("Values")

    ax = plt.gca()
    ax.yaxis.grid(False)
    ax.xaxis.grid(True)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick2On = False
        tick.label2On = False
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False
        tick.label2On = False

    axR = fig.add_subplot(111, sharex=axL, frameon=False)
    line2, = axR.plot(result, "r-", alpha=.7)
    axR.yaxis.tick_right()
    axR.yaxis.set_label_position("right")
    plt.ylabel("Score")
    plt.ylim(ymin=-5.0)
    plt.xlabel("Sample data")

    ax = plt.gca()
    ax.yaxis.grid(False)
    ax.xaxis.grid(False)
    for tick in ax.yaxis.get_major_ticks():
        tick.tick2On = True
        tick.label2On = True
    for tick in ax.xaxis.get_major_ticks():
        tick.tick2On = False
        tick.label2On = False

    plt.title("Sample: Change Anomaly Detection")
    plt.legend([line1, line2], ["Data", "Score"], loc=2)
    plt.savefig("sample.png", dpi=144)