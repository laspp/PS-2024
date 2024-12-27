//#include <cuda.h>
#ifdef __cplusplus
extern "C" {
#endif

__global__ void vectorSubtract(float *c, const float *a, const float *b, int len) {
	int gid = threadIdx.x;	
	c[gid] = a[gid] - b[gid];
}

#ifdef __cplusplus
}
#endif
