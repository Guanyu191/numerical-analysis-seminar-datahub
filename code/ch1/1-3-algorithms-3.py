import numpy as np

def polygonarea(x, y):
    n = len(x)
    area = 0
    for k in range(n):
        k_next = (k + 1) % n
        area += x[k] * y[k_next] - x[k_next] * y[k]
    area = 0.5 * abs(area)
    return area

if __name__ == "__main__":
    # square (0,0), (0,1), (1,1), (1,0)
    x_square = np.array([0, 0, 1, 1])
    y_square = np.array([0, 1, 1, 0])
    square_area = polygonarea(x_square, y_square)
    print("area of square:", square_area)

    # equilateral triangle (0, 0), (1, 0), (0.5, np.sqrt(3) / 2)
    x_triangle = np.array([0, 1, 0.5])
    y_triangle = np.array([0, 0, np.sqrt(3) / 2])
    triangle_area = polygonarea(x_triangle, y_triangle)
    print("area of equilateral triangle:", triangle_area)
