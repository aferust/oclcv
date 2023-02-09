
#if defined (ERODE)
    #define COMPARE(a, b) (min((a), (b)))
    #define VALUE 255
#else
    #define COMPARE(a, b) (max((a), (b)))
    #define VALUE 0
#endif


// https://github.com/mompes/CUDA-dilation-and-erosion-filters
//https://towardsdatascience.com/get-started-with-gpu-image-processing-15e34b787480

__kernel void morphed(
	__global const uchar* src,
	__global uchar* dst,
	const int2 size,
    const int _radio)
{
	const int2 gid = { get_global_id(0), get_global_id(1) };
	
	if(!all(gid < size))
		return;
		
	const int gid1 = gid.x + gid.y * size.x;

    const int x = gid.x; const int width = size.x;
    const int y = gid.y; const int height = size.y;

    const unsigned int start_i = max(y - _radio, 0);
    const unsigned int end_i = min(height - 1, y + _radio);
    const unsigned int start_j = max(x - _radio, 0);
    const unsigned int end_j = min(width - 1, x + _radio);
    uchar value = VALUE;
    for (int i = start_i; i <= end_i; i++) {
        for (int j = start_j; j <= end_j; j++) {
            value = COMPARE(value, src[i * width + j]); // min for erode
        }
    }
    dst[gid1] = value; 
}

__kernel void morphSharedStep2(__global const uchar* src, __global uchar* dst, 
                                 /*int radio,*/ int width, int height, int tile_w, int tile_h) {
    __local uchar smem[SHARED_SIZE];
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int x = bx * tile_w + tx;
    int y = by * tile_h + ty - radio;

    int blockDim_x = get_local_size(0);

    smem[ty * blockDim_x + tx] = VALUE;
    barrier(CLK_LOCAL_MEM_FENCE); 
    if (x >= width || y < 0 || y >= height) {
        return;
    }
    smem[ty * blockDim_x + tx] = src[y * width + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (y < (by * tile_h) || y >= ((by + 1) * tile_h)) {
        return;
    }
    __local uchar* smem_thread = &smem[(ty - radio) * blockDim_x + tx];
    uchar val = smem_thread[0];
#pragma unroll
    for (int yy = 1; yy <= 2 * radio; yy++) {
        val = COMPARE(val, smem_thread[yy * blockDim_x]);
    }
    dst[y * width + x] = val;
}

__kernel void morphSharedStep1(__global const uchar* src, __global uchar* dst, 
            /*int radio,*/ int width, int height, int tile_w, int tile_h) {
    
    __local uchar smem[SHARED_SIZE];
    int tx = get_local_id(0);
    int ty = get_local_id(1);
    int bx = get_group_id(0);
    int by = get_group_id(1);
    int x = bx * tile_w + tx - radio;
    int y = by * tile_h + ty;

    int blockDim_x = get_local_size(0);

    smem[ty * blockDim_x + tx] = VALUE;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x < 0 || x >= width || y >= height) {
        return;
    }
    smem[ty * blockDim_x + tx] = src[y * width + x];
    barrier(CLK_LOCAL_MEM_FENCE);
    if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w)) {
        return;
    }
    __local uchar* smem_thread = &smem[ty * blockDim_x + tx - radio];
    uchar val = smem_thread[0];

#pragma unroll
    for (int xx = 1; xx <= 2 * radio; xx++) {
        val = COMPARE(val, smem_thread[xx]);
    }
    dst[y * width + x] = val;
}