#and gate : MCP Neuron

def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    a = w1 * x1 + w2 * x2

    if a < theta:
        return 0
    else:
        return 1

y1 = np.array([0, 0])
print(y1)