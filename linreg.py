import numpy as np
x=np.array([1,2,3,4,5])
y=np.array([2,4,6,8,14])
xmean=np.mean(x)
ymean=np.mean(y)
y1=0
x1=0
for i in range(len(x)):
  y1+=(y[i]-ymean)*(x[i]-xmean)
  x1+=(x[i]-xmean)**2
m=y1/x1
b=ymean-m*xmean
print(m)
print(b)
pred_y=m*x[4]+b
print(pred_y)
print(y[4])
