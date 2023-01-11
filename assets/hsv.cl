

__kernel void rgb2hsv(
	__global const uchar* src_rgb,
	__global uchar* dst_hsv,
	const int2 size)
{
    const int2 gid = { get_global_id(0), get_global_id(1) };

    if(!all(gid < size))
		return;
    
    const int gid1 = gid.x + gid.y * size.x;

    float r = (float)(src_rgb[3*gid1 + 0]) * (1.0f / 255.0f);
    float g = (float)(src_rgb[3*gid1 + 1]) * (1.0f / 255.0f);
    float b = (float)(src_rgb[3*gid1 + 2]) * (1.0f / 255.0f);

    float cmax = fmax(r, fmax(g, b));
    float cmin = fmin(r, fmin(g, b));
    float cdelta = cmax - cmin;

    uchar h = (uchar)((cdelta == 0) ? 0 : (cmax == r) ? 60.0f * ((g - b) / cdelta) : (cmax == g)
            ? 60.0f * ((b - r) / cdelta + 2) : 60.0f * ((r - g) / cdelta + 4));

    if (h < 0)
        h += 360;
    
    h /= 2.0f; // map Hue between 0-180

    auto s = (uchar)(100.0f * (cmax == 0 ? 0 : cdelta / cmax));
    auto v = (uchar)(100.0f * cmax);

    dst_hsv[3*gid1 + 0] = h;
    dst_hsv[3*gid1 + 1] = s;
    dst_hsv[3*gid1 + 2] = v;

}