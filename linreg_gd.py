m = 0
b = 0
alpha = 0.01

x = [1, 2, 3]
y = [2, 4, 6]
n = len(x)

for i in range(1000):
    loss_m = 0
    loss_b = 0

    for j in range(n):
        y_pred = m*x[j] + b
        loss_m += -2*x[j]*(y[j] - y_pred)
        loss_b += -2*(y[j] - y_pred)

    m = m - alpha*(loss_m/n)
    b = b - alpha*(loss_b/n)

print(m, b)