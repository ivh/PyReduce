all: clib/_cluster.so clib/_extract.so

clib/_person.so: clib/build_cluster.py
    python $<

clib/_fnmatch.so: clib/build_extract.py
    python $<

clean:
rm clib/_*.c clib/_*.o clib/_*.so

