objects= main.o gpu.o test.o
NVCC= nvcc
# opt= -g -G -DEBUG -D_DEBUG
opt= -O2 -g -G
LIBS=  
execname= main.out


#compile
$(execname): $(objects)
	$(NVCC) $(opt) -o $(execname) $(objects) $(LIBS) 

gpu.o: gpu.cu
	$(NVCC) $(opt) $(ARCH) -c gpu.cu
test.o: test.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c test.cu
main.o: main.cu
	$(NVCC) $(opt) $(ARCH) -std=c++11 -c main.cu

clean:
	rm $(objects)
