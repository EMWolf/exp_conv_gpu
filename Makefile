CC       = nvcc

CCFLAGS  = ###-lm -I../common

BIN =  test_cuda

all: $(BIN)


exp_conv_cuda: exp_conv_cuda.cu
	
	$(CC) $(CCFLAGS) $< -o  $@


clean:

	$(RM) $(BIN)