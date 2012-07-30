_C_gdt.so: C_gdt.o
	gcc -O3 -shared -o _C_gdt.so C_gdt.o

C_gdt.o: C_gdt.c C_gdt.h
	gcc -std=c99 -O3 -funroll-loops -ffast-math -fPIC -c C_gdt.c -I$(HOME)/local/include/python2.7 -I$(HOME)/local/lib/python2.7/site-packages/numpy/core/include/numpy

.PHONY: clean

clean:
	rm -f _C_gdt.so C_gdt.o
