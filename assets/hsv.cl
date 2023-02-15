

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

    float _h = (float)((cdelta == 0) ? 0 : (cmax == r) ? 60.0f * ((g - b) / cdelta) : (cmax == g)
            ? 60.0f * ((b - r) / cdelta + 2) : 60.0f * ((r - g) / cdelta + 4));

    if (_h < 0)
        _h += 360.0f;
    
    _h /= 2.0f; // map Hue between 0-180
    uchar h = (uchar)_h;
    uchar s = (uchar)(100.0f * (cmax == 0 ? 0 : cdelta / cmax));
    uchar v = (uchar)(100.0f * cmax);

    dst_hsv[3*gid1 + 0] = h;
    dst_hsv[3*gid1 + 1] = s;
    dst_hsv[3*gid1 + 2] = v;

}

__kernel void hsv2rgb(
    __global const uchar* src_hsv,
    __global uchar* dst_rgb,
    const int2 size)
{
    const int2 gid = { get_global_id(0), get_global_id(1) };

    if(!all(gid < size))
        return;
    
    const int gid1 = gid.x + gid.y * size.x;

    float h = ((float)src_hsv[3*gid1 + 0]) * 2.0f;
    float s = ((float)src_hsv[3*gid1 + 1]) * 0.01f;
    float v = ((float)src_hsv[3*gid1 + 2]) * 0.01f;

    if (s <= 0)
    {
        dst_rgb[3*gid1 + 0] = (uchar)(v * 255);
        dst_rgb[3*gid1 + 1] = (uchar)(v * 255);
        dst_rgb[3*gid1 + 2] = (uchar)(v * 255);
        return;
    }

    if (v <= 0.0f)
    {
        dst_rgb[3*gid1 + 0] = 0;
        dst_rgb[3*gid1 + 1] = 0;
        dst_rgb[3*gid1 + 2] = 0;
        return;
    }

    if (h >= 360.0f)
        h = 0.0f;
    else
        h /= 60.0f;

    int hh = (int)h;
    float ff = h - (float)hh;

    float p = v * (1.0f - s);
    float q = v * (1.0f - (s * ff));
    float t = v * (1.0f - (s * (1.0f - ff)));

    float r;
    float g;
    float b;

    switch (hh)
    {
    case 0:
        r = v;
        g = t;
        b = p;
        break;
    case 1:
        r = q;
        g = v;
        b = p;
        break;
    case 2:
        r = p;
        g = v;
        b = t;
        break;
    case 3:
        r = p;
        g = q;
        b = v;
        break;
    case 4:
        r = t;
        g = p;
        b = v;
        break;
    case 5:
    default:
        r = v;
        g = p;
        b = q;
        break;
    }

    dst_rgb[3*gid1 + 0] = (uchar)(r * 255);
    dst_rgb[3*gid1 + 1] = (uchar)(g * 255);
    dst_rgb[3*gid1 + 2] = (uchar)(b * 255);

}