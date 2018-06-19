all: clib/_cluster.so clib/_extract.so

clib/_cluster.so: clib/build_cluster.py
	python3 $<

clib/_extract.so: clib/build_extract.py
	python3 $<

clean:
	rm clib/_*.c clib/_*.o clib/_*.so

