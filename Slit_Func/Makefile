CC = gcc
CFLAGS = -Ofast -Wall -lm -lcplcore

binaries = slit_func.bin

$(binaries): %.bin: %.c
	$(CC) $< -o $@ $(CFLAGS)

all: $(binaries)

slit_func_lapack.bin: %.bin: %.c
	icc $< -o $@ -lm -mkl    
