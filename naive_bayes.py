# ライブラリの定義
import numpy as np
import matplotlib.pyplot as plt
import time

class naiveBayes:
    def __init__(self, databaseCSV, testCSV, pathlossdata):
        # fingerprint
        data = np.loadtxt(databaseCSV, delimiter=',', skiprows=1, encoding="utf-8_sig")  # データ読み込み
        self.XY   = data[:,0:2]                                         # X,Y値
        self.RSSI = np.array([data[:,2], data[:,3], data[:,4]])

        # test 
        test = np.loadtxt(testCSV, delimiter=',', skiprows=1, encoding="utf-8_sig")  # データ読み込み
        self.test_XY   = test[:,1:3]                                    # X,Y値
        self.test_RSSI = np.array([test[:,3], test[:,4], test[:,5]])

        # variance of Pathloss
        self.var = 0
        for i in range(18):
            pldata = []
            with open(pathlossdata + '/' + str(i+1) + '.txt' ) as f:
                for line in f:
                    if line == '\n':
                        continue
                    pldata.append(-1 * int(line.split()[-1]))
            self.var += np.var(pldata)
        self.var = self.var/18

        # estimation result
        self.est = np.zeros(len(self.test_XY)) 

    def plotInput(self):
        ### Plot input data
        plt.plot(self.XY[:, 0],self.XY[:, 1],'^')
        plt.plot(self.test_XY[:, 0],self.test_XY[:, 1],'o')
        plt.grid(True)
        plt.show()

    def estimate(self):
        result = np.zeros((len(self.test_XY), len(self.XY), 4))   # The results of bayes theorem
        for k, [RSSi, test_RSSi] in enumerate(zip(self.RSSI, self.test_RSSI)):
            for i in range(len(self.test_XY)):
                for j in range(len(self.XY)):
                    priorProbability = np.count_nonzero(RSSi == RSSi[j]) / len(RSSi)
                    var = self.var
                    result[i,j,k] = np.exp(-np.square(RSSi[j] - test_RSSi[i])/ 2 / var)/ np.sqrt(2 * np.pi * var) # 分散＝１に固定しているので改良の余地あり
                    result[i,j,k] = result[i,j,k] * priorProbability    # ベイズの分子
                result[i,:,k] = result[i,:,k] / np.sum(result[i,:,k])   # ベイズの定理
            result[:,:,3] += result[:,:,k]

        self.est = np.argmax(result[:,:,3]/len(self.RSSI) ,axis=1)   # Estimation of most likly fingerprint

    def showResults(self):
        dis = (self.test_XY - self.XY[self.est])**2
        SE = np.sqrt(dis[:,0] + dis[:,1])   # squared error
        MSE = np.mean(SE)                   # mean squared error
        VSE = np.var(SE)                    # variance squared error

        print(MSE, VSE) # 2.3834, 0.8061

        ### Plot CDF data
        x_label, const = np.unique(np.sort(SE), return_counts=True)
        CDF = const
        for n in range(len(const)):
            if n!=0:
                CDF[n] += CDF[n-1]

        # plt.plot(x_label, CDF/10, linestyle='-', linewidth=1, markersize=3)
        # plt.grid(True)
        # plt.show()

        return x_label, CDF/len(self.test_XY)


if __name__ == '__main__':
    wls = ['ble', 'wifi', 'zigbee']
    for wl in wls:
        S1 = naiveBayes('Scenario1/Database_S1_' + wl + '.csv', 'Scenario1/Tests_S1_' + wl + '.csv', 'Scenario1/Pathloss/' + wl)
        # S1.plotInput()
        S1.estimate()
        x,y = S1.showResults()
        plt.plot(x, y, linestyle='dotted', linewidth=1, markersize=3)
    plt.grid(True)
    plt.show()
