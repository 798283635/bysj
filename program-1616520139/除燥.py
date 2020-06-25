import matplotlib.pyplot as plt
import pywt
import pandas as  pd
# Get data:
dataframe = pd.read_csv('six.csv', sep=',')

dataset = dataframe["hourly_traffic_count"].values

ecg = dataset

index = []
data = []
for i in range(len(ecg)-1):
    X = float(i)
    Y = float(ecg[i])
    index.append(X)
    data.append(Y)

print(pywt.wavelist())
# Create wavelet object and define parameters
w = pywt.Wavelet('sym17')  # 选用Daubechies8小波
maxlev = pywt.dwt_max_level(len(data), w.dec_len)
print("maximum level is " + str(maxlev))
#threshold = 0.05  # Threshold for filtering
threshold = 0.2
# Decompose into wavelet components, to the level selected:
coeffs = pywt.wavedec(data, 'sym17', level=maxlev)  # 将信号进行小波分解

plt.figure()
for i in range(1, len(coeffs)):
    coeffs[i] = pywt.threshold(coeffs[i], threshold*max(coeffs[i]))  # 将噪声滤波

datarec = pywt.waverec(coeffs, 'sym17')  # 将信号进行小波重构
print(ecg)
print(datarec)



mintime = 0
maxtime = mintime + len(data) + 1

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(index[mintime:maxtime], data[mintime:maxtime])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("Raw signal")
plt.subplot(2, 1, 2)
plt.plot(index[mintime:maxtime], datarec[mintime:maxtime-1])
plt.xlabel('time (s)')
plt.ylabel('microvolts (uV)')
plt.title("De-noised signal using wavelet techniques")


plt.tight_layout()
plt.show()

dataframe["hourly_traffic_count"] = datarec

dataframe.to_csv("after_six1.csv",index = False)
#plt.plot(ecg[0:400])
plt.plot(datarec[0:400],c='r')
plt.show()