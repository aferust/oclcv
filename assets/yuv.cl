

__kernel void yuv2rgb(
	__global const uchar* src_yuv,
	__global uchar* dst_rgb,
	const int2 size)
{
    const int2 gid = { get_global_id(0), get_global_id(1) };

    if(!all(gid < size))
		return;
    
    const int gid1 = gid.x + gid.y * size.x;

	const int y = src_yuv[3*gid1 + 0] - 16;
	const int u = src_yuv[3*gid1 + 1] - 128;
	const int v = src_yuv[3*gid1 + 2] - 128;

    dst_rgb[3*gid1 + 0] = clamp((298 * y + 409 * v + 128) >> 8, 0, 255);
    dst_rgb[3*gid1 + 1] = clamp((298 * y - 100 * u - 208 * v + 128) >> 8, 0, 255);
    dst_rgb[3*gid1 + 2] = clamp((298 * y + 516 * u + 128) >> 8, 0, 255);
}