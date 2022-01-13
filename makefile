CC=g++
CFLAGS=-I.
DEPS = matrix.h multiply.h

%.o: %.c $(DEPS)
	$(CC) -c -o $@ $< $(CFLAGS)

matmul: matrix.o main.o
	$(CC) -o matmul matmul.o main.o