CC = gcc
CFLAGS = -lm -lcplcore

binaries = slit_func.bin

$(binaries): %.bin: %.c
	$(CC) $(CFLAGS) $< -o $@

all: $(binaries)
    
