def poly1(p, x=-1):
    result = 0
    for i, coefficient in enumerate(p):
        result += coefficient * (x) ** i
    return result

if __name__ == '__main__':
    r = [1, -1, 0, 3]
    print(poly1(r))

    s = [0, -1, 2]
    print(poly1(s))
