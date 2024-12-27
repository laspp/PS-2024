//#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void Hello(void) {
	printf("Hello from thread %d.%d!\n", blockIdx.x, threadIdx.x);
}

#ifdef __cplusplus
}
#endif
