import matplotlib.pyplot as plt

with open('fixed_data.txt', 'r') as f:
    lines = f.readlines()

lines = [l.split(',') for l in lines]
lines = lines[1:]

vanilla16 = []
vanilla32 = []
cache_block16 = []
cache_block32 = []
sse16 = []
sse32 = []

for line in lines:
    if line[1] == 'vanilla16':
        vanilla16.append(line)
    elif line[1] == 'vanilla32':
        vanilla32.append(line)
    elif line[1] == 'cacheblock16':
        cache_block16.append(line)
    elif line[1] == 'cacheblock32':
        cache_block32.append(line)
    elif line[1] == 'sse16':
        sse16.append(line)
    elif line[1] == 'sse32':
        sse32.append(line)
    else:
        print("INVALID METHOD" + line[1])
        exit()

v16 = [int(l[3]) for l in vanilla16]
v32 = [int(l[3]) for l in vanilla32]

c16 = [int(l[3]) for l in cache_block16]
c32 = [int(l[3]) for l in cache_block32]

s16 = [int(l[3]) for l in sse16]
s32 = [int(l[3]) for l in sse32]

plt.plot(range(10, len(v16) + 10), v16, label="Vanilla (16-Bit)", linewidth=4)
plt.plot(range(10, len(v32) + 10), v32, label="Vanilla (32-Bit)", linewidth=4)

plt.plot(range(10, len(c16) + 10), c16, label="Cache-Aware (16-Bit)", linewidth=4)
plt.plot(range(10, len(c32) + 10), c32, label="Cache-Aware (32-Bit)", linewidth=4)

plt.plot(range(10, len(s16) + 10), s16, label="SSE SIMD (16-Bit)", linewidth=4)
plt.plot(range(10, len(s32) + 10), s32, label="SSE SIMD (32-Bit)", linewidth=4)

plt.legend()
plt.xlabel("Matrix Size (Square)")
plt.ylabel("Microseconds to Multiply (100 Trial Avg)")
plt.title("Integer Matrix Multiplication Algorithms")

plt.savefig("res/performance_fixed.png", dpi=300) #save as png
plt.show()


