import numpy as np

class BinaryQubit:
    def __init__(self, states, weights):
        self.states = np.array(states, dtype=np.int32)
        self.weights = np.array(weights, dtype=np.float64)
        self.normalize()

    def normalize(self):
        norm = np.sqrt(np.sum(self.weights ** 2))
        if norm != 0:
            self.weights = self.weights / norm

    def apply_gate(self, matrix):
        self.weights = matrix @ self.weights
        self.normalize()

    def apply_superposition(self, kernel_fn):
        new_weights = np.zeros_like(self.weights)
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                similarity = kernel_fn(self.states[i], self.states[j])
                new_weights[i] += similarity * self.weights[j]
        self.weights = new_weights
        self.normalize()

    def measure(self):
        return self.states[np.argmax(self.weights)]

def hadamard(n):
    H = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    result = H
    for _ in range(n - 1):
        result = np.kron(result, H)
    return result

def not_gate(n, target_bit):
    size = 2 ** n
    matrix = np.eye(size)
    for i in range(size):
        flipped = i ^ (1 << (n - target_bit - 1))
        matrix[i], matrix[flipped] = matrix[flipped], matrix[i].copy()
    return matrix

def hamming_kernel(x, y, tau=1.0):
    hamming = np.sum(np.bitwise_xor(x, y))
    return np.exp(-hamming / tau)

# Пример запуска
if __name__ == "__main__":
    states = [[0, 0], [0, 1], [1, 0], [1, 1]]
    weights = [0.5, 0.5, 0.0, 0.0]
    q = BinaryQubit(states, weights)
    print("Initial state:", q.states.tolist(), q.weights.tolist())
    q.apply_gate(hadamard(2))
    print("After Hadamard:", q.weights.tolist())
    q.apply_superposition(hamming_kernel)
    print("After Superposition:", q.weights.tolist())
    print("Measured:", q.measure().tolist())
