//#include <cuda.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void vectorSubtract(float *c, const float *a, const float *b, int len) {
	// določimo globalni indeks elementov
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	// če je niti manj kot je dolžina vektorjev, morajo nekatere izračunati več razlik
	while (gid < len) {
		c[gid] = a[gid] - b[gid];
		gid += gridDim.x * blockDim.x;
	}
}

#ifdef __cplusplus
}
#endif
