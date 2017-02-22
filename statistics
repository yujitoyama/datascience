from collections import Counter
from matplotlib import pyplot as plt
import csv
import numpy as np
import math as mt
import random
from sympy import Symbol,exp,sqrt,pi,Integral


def sum_of_squares(v):
    return np.dot(v,v)

#csv読み込み
def read_csv(filename):
    friendscount = []
    times = []
    with open(filename) as f:
        reader = csv.reader(f)
        next(f)
        for row in reader:
            friendscount.append(int(row[1]))
            times.append(int(row[2]))
    return friendscount,times

#散布図
def drawscatter(x,y):
    plt.scatter(x,y)
    plt.axis('equal')
    plt.xlabel('# of friends')
    plt.ylabel('minutes per day')
    plt.title('Correlation with an Outlier')
    plt.show()

#平均値
def mean(x):
    return sum(x) / len(x)

#中央値
def median(v):
    n = len(v)
    sorted_v = sorted(v)
    midpoint = n // 2

    if n % 2 == 1:
        return sorted_v[midpoint]
    else:
        lo = midpoint -1
        hi = midpoint 
        return (sorted_v[lo] + sorted_v[hi])/2

#分位数
def quantile(x,p):
    p_index = int(p * len(x))
    return sorted(x)[p_index]

#最頻値
def mode(x):
    counts = Counter(x)
    max_count = max(counts.values())
    return [x_i for x_i, count in counts if count == max_count]

#散らばり
def data_range(x):
    return max(x) - min(x)

def de_mean(x):
    x_bar = mean(x)
    return [x_i - x_bar for x_i in x]

#分散
def variance(x):
    n = len(x)
    deviations = de_mean(x)
    return sum_of_squares(deviations)/n-1

#標準偏差
def standard_deviation(x):
    return mt.sqrt(variance(x))

#共分散
def covariance(x,y):
    n = len(x)
    return np.dot(de_mean(x),de_mean(y))/(n-1)

#相関係数
def correlation(x,y):
    stddev_x = standard_deviation(x)
    stddev_y = standard_deviation(y)
    if stddev_x > 0 and stddev_y >0:
        return covariance(x,y) / stddev_x / stddev_y
    else:
        return 0

#確率：家族構成シミュレーション
def random_kid():
    return random.choice(['boy','girl'])

#確率：家族構成シミュレーション２
def checkpossibility():
    both_girls = 0
    older_girl = 0
    either_girl = 0

    random.seed(0)
    for _ in range(10000):
        younger = random_kid()
        older = random_kid()
        if older == 'girl':
            older_girl += 1
        if older == 'girl' and younger == 'girl':
            both_girls += 1
        if older == 'girl' or younger == 'girl':
            either_girl += 1
    
    print('P(both girl | younger girl):',both_girls / older_girl)
    print('P(both girl | either girl):',both_girls / either_girl)

#確率：確率密度関数（一様分布）
def uniform_pdf(x):
    return 1 if x >= 0 and x < 1 else 0

#確率：積分器
def integral(fx,x,a,b):
    x = Symbol('x')
    result = Integral(fx,(x,a,b)).doit().evalf()
    print(result)

#確率：正規分布
def normal_pdf(x,mu=0,sigma=1):
    sqrt_two_pi = mt.sqrt(2 * mt.pi)
    return (mt.exp(-(x-mu)**2/2/sigma**2)/sqrt(sqrt_two_pi*sigma))

#確率：正規分布の記述
def drawnomal_pdf():
    xs = [x/10 for x in range(-50,50)]
    plt.plot(xs,[normal_pdf(x,sigma=1) for x in xs],'-',label='mu=0,sigma=1')
    plt.plot(xs,[normal_pdf(x,sigma=2) for x in xs],'--',label='mu=0,sigma=2')
    plt.legend()
    plt.axis([-50,50,0,1])
#軸の自動調整    plt.axis('scaled')
#軸単位の手動調整    plt.xlim([-50,50])
    plt.xlabel('x')
    plt.ylabel('value of normal_pdf')
    plt.title('various normal pdfs')
    plt.show()
 
if __name__ == '__main__':
    drawnomal_pdf()

'''
    friendscounts,times = read_csv('fcti.csv')
    drawscatter(friendscounts,times)
    print(correlation(friendscounts,times))
    checkpossibility()
'''
