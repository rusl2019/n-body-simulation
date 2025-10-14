all:
	nvcc main.cu -o cuda_sim -lGL -lGLEW -lglfw -Xcompiler "-fno-gnu-unique"
