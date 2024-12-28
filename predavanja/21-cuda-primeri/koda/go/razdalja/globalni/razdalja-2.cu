// nadgrajen ščepec vectorSubtract4
// računanju razlike je dodano še kvadriranje

#ifdef __cplusplus
extern "C" {
#endif

__global__ void vectorDistance2(float *c, const float *a, const float *b, int len) {
	// določimo globalni indeks elementov
	int gid = blockIdx.x * blockDim.x + threadIdx.x;
	float diff;
	// če je niti manj kot je dolžina vektorjev, morajo nekatere izračunati več razlik
	while (gid < len) {
		diff = a[gid] - b[gid];
		c[gid] = diff * diff;
		gid += gridDim.x * blockDim.x;
	}
}

#ifdef __cplusplus
}
#endif
