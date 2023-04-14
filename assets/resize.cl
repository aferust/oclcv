// reimplementation of https://github.com/vencabkk/opencl-resizer/blob/master/OpenCL/kernels/resize.cl

#ifdef cl_amd_fp64
#pragma OPENCL EXTENSION cl_amd_fp64:enable
#elif defined (cl_khr_fp64)
#pragma OPENCL EXTENSION cl_khr_fp64:enable
#endif

inline float4 sample_bicubic_border(__global const uchar* source, float2 pos, int2 SrcSize)
{
   int2 isrcpos = convert_int2(pos);
   float dx = pos.x - isrcpos.x;
   float dy = pos.y - isrcpos.y;

   float4 C[4] = {0, 0, 0, 0};

   if (isrcpos.x < 0 || isrcpos.x >= SrcSize.x)
      return 0;

   if (isrcpos.y < 0 || isrcpos.y >= SrcSize.y)
      return 0;

   #pragma unroll
   for (int i = 0; i < 4; i++)
   {
      int y = isrcpos.y - 1 + i;
      if (y < 0)
         y = 0;

      if (y >= SrcSize.y)
         y = SrcSize.y - 1;

      int Middle = clamp(isrcpos.x, 0, SrcSize.x - 1);

      const int2 pos0 = { Middle, y };
      const int pos0gid = pos0.x + pos0.y * SrcSize.x;

      float4 center = (float4){(float)source[3*pos0gid + 0], (float)source[3*pos0gid + 1], (float)source[3*pos0gid + 2], 0.0f};

      float4 left = 0, right1 = 0, right2 = 0;
      if (isrcpos.x - 1 >= 0)
      {
         const int2 pos1 = { isrcpos.x - 1, y };
         const int pos1gid = pos1.x + pos1.y * SrcSize.x;
         left = (float4){(float)source[3*pos1gid + 0], (float)source[3*pos1gid + 1], (float)source[3*pos1gid + 2], 0.0f};
      }
      else
      {
         left = center;
      }

      if (isrcpos.x + 1 < SrcSize.x)
      {
         const int2 pos2 = { isrcpos.x + 1, y };
         const int pos2gid = pos2.x + pos2.y * SrcSize.x;
         right1 = (float4){(float)source[3*pos2gid + 0], (float)source[3*pos2gid + 1], (float)source[3*pos2gid + 2], 0.0f};
      }
      else
      {
         right1 = center;
      }

      if (isrcpos.x + 2 < SrcSize.x)
      {
         const int2 pos3 = { isrcpos.x + 2, y };
         const int pos3gid = pos3.x + pos3.y * SrcSize.x;
         right2 = (float4){(float)source[3*pos3gid + 0], (float)source[3*pos3gid + 1], (float)source[3*pos3gid + 2], 0.0f};
      }
      else
      {
         right2 = right1;
      }

      float4 a0 = center;
      float4 d0 = left - a0;
      float4 d2 = right1 - a0;
      float4 d3 = right2 - a0;

      float4 a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
      float4 a2 =  1.0f / 2 * d0 + 1.0f / 2 * d2;
      float4 a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
      C[i] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;
   }

   float4 d0 = C[0] - C[1];
   float4 d2 = C[2] - C[1];
   float4 d3 = C[3] - C[1];
   float4 a0 = C[1];
   float4 a1 = -1.0f / 3 * d0 + d2 -1.0f / 6 * d3;
   float4 a2 = 1.0f / 2 * d0 + 1.0f / 2 * d2;
   float4 a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;

   return a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
}

__kernel void resize_bicubic(__global const uchar* source, __global uchar* dest, 
    const int2 SrcSize, const int2 dstSize)
{
   float ratioX = (float)dstSize.x / (float)SrcSize.x;
   float ratioY = (float)dstSize.y / (float)SrcSize.y;
   
   const int2 pos = { get_global_id(0), get_global_id(1) };

   if(!all(pos < dstSize))
      return;

   float2 srcpos = {((float)pos.x + 0.4995f) / ratioX, ((float)pos.y + 0.4995f) / ratioY};

   float4 value;

   srcpos -= (float2)(0.5f, 0.5f);

   int2 isrcpos = convert_int2(srcpos);
   float dx = srcpos.x - isrcpos.x;
   float dy = srcpos.y - isrcpos.y;

   if (isrcpos.x <= 0 || isrcpos.x >= SrcSize.x - 2)
      value = sample_bicubic_border(source, srcpos, SrcSize);

   if (isrcpos.y <= 0 || isrcpos.y >= SrcSize.y - 2)
      value = sample_bicubic_border(source, srcpos, SrcSize);

   float4 C[4] = {0, 0, 0, 0};

   #pragma unroll
   for (int i = 0; i < 4; i++)
   {
      const int y = isrcpos.y - 1 + i;

      int x1 = (int)(srcpos.x);
      int x2 = (int)(srcpos.x + 1);
      int y1 = (int)(srcpos.y);
      int y2 = (int)(srcpos.y + 1);

      const int2 pos0 = { x1, y1 };
      const int2 pos1 = { x2, y1 };
      const int2 pos2 = { x1, y2 };
      const int2 pos3 = { x2, y2 };

      const int pos0gid = pos0.x + pos0.y * SrcSize.x;
      const int pos1gid = pos1.x + pos1.y * SrcSize.x;
      const int pos2gid = pos2.x + pos2.y * SrcSize.x;
      const int pos3gid = pos3.x + pos3.y * SrcSize.x;
      
      float4 a0 = (float4){(float)source[3*pos0gid + 0], (float)source[3*pos0gid + 1], (float)source[3*pos0gid + 2], 0.0f};
      float4 d0 = (float4){(float)source[3*pos1gid + 0], (float)source[3*pos1gid + 1], (float)source[3*pos1gid + 2], 0.0f} - a0;
      float4 d2 = (float4){(float)source[3*pos2gid + 0], (float)source[3*pos2gid + 1], (float)source[3*pos2gid + 2], 0.0f} - a0;
      float4 d3 = (float4){(float)source[3*pos3gid + 0], (float)source[3*pos3gid + 1], (float)source[3*pos3gid + 2], 0.0f} - a0;

      float4 a1 = -1.0f / 3 * d0 + d2 - 1.0f / 6 * d3;
      float4 a2 =  1.0f / 2 * d0 + 1.0f / 2 * d2;
      float4 a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
      C[i] = a0 + a1 * dx + a2 * dx * dx + a3 * dx * dx * dx;
   }

   float4 d0 = C[0] - C[1];
   float4 d2 = C[2] - C[1];
   float4 d3 = C[3] - C[1];
   float4 a0 = C[1];
   float4 a1 = -1.0f / 3 * d0 + d2 -1.0f / 6 * d3;
   float4 a2 = 1.0f / 2 * d0 + 1.0f / 2 * d2;
   float4 a3 = -1.0f / 6 * d0 - 1.0f / 2 * d2 + 1.0f / 6 * d3;
   value = a0 + a1 * dy + a2 * dy * dy + a3 * dy * dy * dy;
    
   const int gid1 = pos.x + pos.y * dstSize.x;
   dest[3*gid1 + 0] = convert_uchar_sat(value.x);
   dest[3*gid1 + 1] = convert_uchar_sat(value.y);
   dest[3*gid1 + 2] = convert_uchar_sat(value.z);
}

__kernel void resize_linear(__global const uchar* source, __global uchar* dest, 
    const int2 SrcSize, const int2 dstSize)
{
   float ratioX = (float)dstSize.x / (float)SrcSize.x;
   float ratioY = (float)dstSize.y / (float)SrcSize.y;
   const int2 pos = { get_global_id(0), get_global_id(1) };

   if(!all(pos < dstSize))
      return;

   float2 srcpos = {((float)pos.x + 0.4995f) / ratioX, ((float)pos.y + 0.4995f) / ratioY};

   float4 value;

   if ((int)(srcpos.x + .5f) == SrcSize.x)
      srcpos.x = SrcSize.x - 0.5001f;

   if ((int)(srcpos.y + .5f) == SrcSize.y)
      srcpos.y = SrcSize.y - 0.5001f;

   srcpos -= (float2)(0.5f, 0.5f);

   if (srcpos.x < -0.5f || srcpos.x >= SrcSize.x - 1 || srcpos.y < -0.5f || srcpos.y >= SrcSize.y - 1)
      value = 0;

   int x1 = (int)(srcpos.x);
   int x2 = (int)(srcpos.x + 1);
   int y1 = (int)(srcpos.y);
   int y2 = (int)(srcpos.y + 1);

   float factorx1 = 1 - (srcpos.x - x1);
   float factorx2 = 1 - factorx1;
   float factory1 = 1 - (srcpos.y - y1);
   float factory2 = 1 - factory1;

   float4 f1 = factorx1 * factory1;
   float4 f2 = factorx2 * factory1;
   float4 f3 = factorx1 * factory2;
   float4 f4 = factorx2 * factory2;

   const int2 pos0 = { x1, y1 };
   const int2 pos1 = { x2, y1 };
   const int2 pos2 = { x1, y2 };
   const int2 pos3 = { x2, y2 };

   const int pos0gid = pos0.x + pos0.y * SrcSize.x;
   const int pos1gid = pos1.x + pos1.y * SrcSize.x;
   const int pos2gid = pos2.x + pos2.y * SrcSize.x;
   const int pos3gid = pos3.x + pos3.y * SrcSize.x;

   float4 v1 = (float4){(float)source[3*pos0gid + 0], (float)source[3*pos0gid + 1], (float)source[3*pos0gid + 2], 0.0f};
   float4 v2 = (float4){(float)source[3*pos1gid + 0], (float)source[3*pos1gid + 1], (float)source[3*pos1gid + 2], 0.0f};
   float4 v3 = (float4){(float)source[3*pos2gid + 0], (float)source[3*pos2gid + 1], (float)source[3*pos2gid + 2], 0.0f};
   float4 v4 = (float4){(float)source[3*pos3gid + 0], (float)source[3*pos3gid + 1], (float)source[3*pos3gid + 2], 0.0f};

   value =  v1 * f1 + v2 * f2 + v3 * f3 + v4 * f4;
    
   const int gid1 = pos.x + pos.y * dstSize.x;
   dest[3*gid1 + 0] = convert_uchar_sat(value.x );
   dest[3*gid1 + 1] = convert_uchar_sat(value.y );
   dest[3*gid1 + 2] = convert_uchar_sat(value.z );
}