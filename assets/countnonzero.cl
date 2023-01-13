// -D OP=?
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

#define COMPARE(a, b) (((a) OP (b)) ? true : false)

void increase(volatile __global ulong* counter){
    atomic_inc(counter);
}

__kernel void countNonZero(__global const uchar* src_mono, uchar cval, __global ulong* ret_val, const int2 size){
    const int2 gid = { get_global_id(0), get_global_id(1) };

    if(!all(gid < size))
		return;
    
    const int gid1 = gid.x + gid.y * size.x;

    volatile __global ulong* counterPtr = ret_val;
    if(COMPARE(src_mono[gid1], cval)) {
        increase(counterPtr);
    }

}