all: clib/_cluster.so clib/_extract.so

clib/_cluster.so: clib/build_cluster.py
	python $<

clib/_extract.so: clib/build_extract.py
	python $<

clean:
	rm clib/_*.c clib/_*.o clib/_*.so

