

__kernel void inRange3(
	__global const uchar* src,
	__global uchar* dst,
	const int2 size,
    const uchar lo0, const uchar hi0,
    const uchar lo1, const uchar hi1,
    const uchar lo2, const uchar hi2,
    const int inverse
    )
{
	const int2 gid = { get_global_id(0), get_global_id(1) };
	
	if(!all(gid < size))
		return;
		
	const int gid1 = gid.x + gid.y * size.x;
    
    const uchar trueVal = (inverse > 0) ? 0 : 255;
    const uchar falseVal = (inverse > 0) ? 255 : 0;

    const uchar c0 = src[3*gid1 + 0];
    const uchar c1 = src[3*gid1 + 1];
    const uchar c2 = src[3*gid1 + 2];

    const bool cond0 = (lo0 < c0) && (c0 < hi0);
    const bool cond1 = (lo1 < c1) && (c1 < hi1);
    const bool cond2 = (lo2 < c2) && (c2 < hi2);

    if(cond0 && cond1 && cond2)
        dst[gid1] = trueVal;
    else
        dst[gid1] = falseVal;
}