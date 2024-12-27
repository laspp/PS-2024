// slaba rešitev: podpora samo za en blok niti

#include <cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vectorSubtract(float *c, const float *a, const float *b, int len) {
	int gid = threadIdx.x;	
	// preprečimo pisanje v nealocirani pomnilnik
	if (gid < len)
		c[gid] = a[gid] - b[gid];
}

#ifdef __cplusplus
}
#endif
