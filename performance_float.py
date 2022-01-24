import matplotlib.pyplot as plt

with open('float_data.txt', 'r') as f:
    lines = f.readlines()

lines = [l.split(',') for l in lines]
lines = lines[1:]

vanilla = []
cache_block = []
sse = []
avx = []
avxfma = []
for line in lines:
    if line[1] == 'vanilla':
        vanilla.append(line)
    elif line[1] == 'cacheblock':
        cache_block.append(line)
    elif line[1] == 'sse':
        sse.append(line)
    elif line[1] == 'avx':
        avx.append(line)
    elif line[1] == 'avxmla':
        avxfma.append(line)
    else:
        print("INVALID METHOD" + line[1])
        exit()

v = [int(l[3]) for l in vanilla]
c = [int(l[3]) for l in cache_block]
s = [int(l[3]) for l in sse]
a = [int(l[3]) for l in avx]
m = [int(l[3]) for l in avxfma]

plt.plot(range(10, len(v) + 10), v, label="Vanilla (Float)", linewidth=4)
plt.plot(range(10, len(c) + 10), c, label="Cache-Aware (Float)", linewidth=4)
plt.plot(range(10, len(s) + 10), s, label="SSE SIMD (Float)", linewidth=4)
plt.plot(range(10, len(a) + 10), a, label="AVX SIMD (Float)", linewidth=4)
plt.plot(range(10, len(m) + 10), m, label="AVX SIMD MLA (Float)", linewidth=4)

plt.legend()
plt.xlabel("Matrix Size (Square)")
plt.ylabel("Microseconds to Multiply (100 Trial Avg)")
plt.title("Floating Point Matrix Multiplication Algorithms")

plt.savefig("res/performance_float.png", dpi=300) #save as png
plt.show()


