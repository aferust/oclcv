//                           License Agreement
//                For Open Source Computer Vision Library
//
// Copyright (C) 2010-2012, Institute Of Software Chinese Academy Of Science,
// all rights reserved. Copyright (C) 2010-2012, Advanced Micro Devices, Inc.,
// all rights reserved. Third party copyrights are property of their respective
// owners.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
//   * Redistribution's of source code must retain the above copyright notice,
//     this list of conditions and the following disclaimer.
//
//   * Redistribution's in binary form must reproduce the above copyright
//   notice,
//     this list of conditions and the following disclaimer in the documentation
//     and/or other materials provided with the distribution.
//
//   * The name of the copyright holders may not be used to endorse or promote
//   products
//     derived from this software without specific prior written permission.
//
// This software is provided by the copyright holders and contributors as is and
// any express or implied warranties, including, but not limited to, the implied
// warranties of merchantability and fitness for a particular purpose are
// disclaimed. In no event shall the Intel Corporation or contributors be liable
// for any direct, indirect, incidental, special, exemplary, or consequential
// damages (including, but not limited to, procurement of substitute goods or
// services; loss of use, data, or profits; or business interruption) however
// caused and on any theory of liability, whether in contract, strict liability,
// or tort (including negligence or otherwise) arising in any way out of
// the use of this software, even if advised of the possibility of such damage.

#if cn != 3
#define loadpix(addr) *(__global const T *)(addr)
#define storepix(val, addr) *(__global T *)(addr) = val
#define TSIZE (int)sizeof(T)
#else
#define loadpix(addr) vload3(0, (__global const T1 *)(addr))
#define storepix(val, addr) vstore3(val, 0, (__global T1 *)(addr))
#define TSIZE (int)sizeof(T1) * cn
#endif

#define OP(a, b)                                                               \
  {                                                                            \
    mid = a;                                                                   \
    a = min(a, b);                                                             \
    b = max(mid, b);                                                           \
  }

#ifdef USE_4OPT

// Utility macros for 1,2,4 channel images:
//   - LOAD4/STORE4 - load/store 4-pixel groups from/to global memory

//  - SHUFFLE4_3/SHUFFLE4_5 - rearrange scattered border/central pixels into
//  regular 4-pixel variables
//      that can be used in following min/max operations

#if cn == 1

#define LOAD4(val, offs)                                                       \
  (val) = vload4(0, (__global T1 *)(srcptr + src_index + (offs)))
#define STORE4(val, offs) vstore4((val), 0, (__global T1 *)(dstptr + (offs)))
#define SHUFFLE4_3(src0, src1, src2, dst0, dst1, dst2)                         \
  {                                                                            \
    dst1 = src1;                                                               \
    dst0 = (T4)(src0, dst1.xyz);                                               \
    dst2 = (T4)(dst1.yzw, src2);                                               \
  }

#define SHUFFLE4_5(src0, src1, src2, src3, src4, dst0, dst1, dst2, dst3, dst4) \
  {                                                                            \
    dst2 = src2;                                                               \
    dst0 = (T4)(src0, src1, dst2.xy);                                          \
    dst1 = (T4)(src1, dst2.xyz);                                               \
    dst3 = (T4)(dst2.yzw, src3);                                               \
    dst4 = (T4)(dst2.zw, src3, src4);                                          \
  }

#elif cn == 2

#define LOAD4(val, offs)                                                       \
  (val) = vload8(0, (__global T1 *)(srcptr + src_index + (offs)))
#define STORE4(val, offs) vstore8((val), 0, (__global T1 *)(dstptr + (offs)))
#define SHUFFLE4_3(src0, src1, src2, dst0, dst1, dst2)                         \
  {                                                                            \
    dst1 = src1;                                                               \
    dst0 = (T4)(src0, dst1.s012345);                                           \
    dst2 = (T4)(dst1.s234567, src2);                                           \
  }

#define SHUFFLE4_5(src0, src1, src2, src3, src4, dst0, dst1, dst2, dst3, dst4) \
  {                                                                            \
    dst2 = src2;                                                               \
    dst0 = (T4)(src0, src1, dst2.s0123);                                       \
    dst1 = (T4)(src1, dst2.s012345);                                           \
    dst3 = (T4)(dst2.s234567, src3);                                           \
    dst4 = (T4)(dst2.s4567, src3, src4);                                       \
  }

#elif cn == 4

#define LOAD4(val, offs)                                                       \
  (val) = vload16(0, (__global T1 *)(srcptr + src_index + (offs)))
#define STORE4(val, offs) vstore16((val), 0, (__global T1 *)(dstptr + (offs)))
#define SHUFFLE4_3(src0, src1, src2, dst0, dst1, dst2)                         \
  {                                                                            \
    dst1 = src1;                                                               \
    dst0 = (T4)(src0, dst1.s0123456789ab);                                     \
    dst2 = (T4)(dst1.s456789abcdef, src2);                                     \
  }

#define SHUFFLE4_5(src0, src1, src2, src3, src4, dst0, dst1, dst2, dst3, dst4) \
  {                                                                            \
    dst2 = src2;                                                               \
    dst0 = (T4)(src0, src1, dst2.s01234567);                                   \
    dst1 = (T4)(src1, dst2.s0123456789ab);                                     \
    dst3 = (T4)(dst2.s456789abcdef, src3);                                     \
    dst4 = (T4)(dst2.s89abcdef, src3, src4);                                   \
  }

#endif

__kernel void medianFilter3_u(__global const uchar *srcptr, int srcStep,
                              int srcOffset, __global uchar *dstptr,
                              int dstStep, int dstOffset, int rows, int cols) {
  int gx = get_global_id(0) << 2;
  int gy = get_global_id(1) << 2;

  if (gy >= rows || gx >= cols)
    return;

  T c0;
  T4 c1;
  T c2;
  T c3;
  T4 c4;
  T c5;
  T c6;
  T4 c7;
  T c8;

  int x_left = mad24(max(gx - 1, 0), TSIZE, srcOffset);
  int x_central = mad24(gx, TSIZE, srcOffset);
  int x_right = mad24(min(gx + 4, cols - 1), TSIZE, srcOffset);

  int xdst = mad24(gx, TSIZE, dstOffset);

  // 0 line
  int src_index = max(gy - 1, 0) * srcStep;
  c0 = *(__global T *)(srcptr + src_index + x_left);
  LOAD4(c1, x_central);
  c2 = *(__global T *)(srcptr + src_index + x_right);

  // 1 line
  src_index = gy * srcStep;
  c3 = *(__global T *)(srcptr + src_index + x_left);
  LOAD4(c4, x_central);
  c5 = *(__global T *)(srcptr + src_index + x_right);

// iteration for one row from 4 row block
#define ITER3(k)                                                               \
  {                                                                            \
    src_index = min(gy + k + 1, rows - 1) * srcStep;                           \
    c6 = *(__global T *)(srcptr + src_index + x_left);                         \
    LOAD4(c7, x_central);                                                      \
    c8 = *(__global T *)(srcptr + src_index + x_right);                        \
    T4 p0, p1, p2, p3, p4, p5, p6, p7, p8;                                     \
    SHUFFLE4_3(c0, c1, c2, p0, p1, p2);                                        \
    SHUFFLE4_3(c3, c4, c5, p3, p4, p5);                                        \
    SHUFFLE4_3(c6, c7, c8, p6, p7, p8);                                        \
    T4 mid;                                                                    \
    OP(p1, p2);                                                                \
    OP(p4, p5);                                                                \
    OP(p7, p8);                                                                \
    OP(p0, p1);                                                                \
    OP(p3, p4);                                                                \
    OP(p6, p7);                                                                \
    OP(p1, p2);                                                                \
    OP(p4, p5);                                                                \
    OP(p7, p8);                                                                \
    OP(p0, p3);                                                                \
    OP(p5, p8);                                                                \
    OP(p4, p7);                                                                \
    OP(p3, p6);                                                                \
    OP(p1, p4);                                                                \
    OP(p2, p5);                                                                \
    OP(p4, p7);                                                                \
    OP(p4, p2);                                                                \
    OP(p6, p4);                                                                \
    OP(p4, p2);                                                                \
    int dst_index = mad24(gy + k, dstStep, xdst);                              \
    STORE4(p4, dst_index);                                                     \
    c0 = c3;                                                                   \
    c1 = c4;                                                                   \
    c2 = c5;                                                                   \
    c3 = c6;                                                                   \
    c4 = c7;                                                                   \
    c5 = c8;                                                                   \
  }

  // loop manually unrolled
  ITER3(0);
  ITER3(1);
  ITER3(2);
  ITER3(3);
}

__kernel void medianFilter5_u(__global const uchar *srcptr, int srcStep,
                              int srcOffset, __global uchar *dstptr,
                              int dstStep, int dstOffset, int rows, int cols) {
  int gx = get_global_id(0) << 2;
  int gy = get_global_id(1) << 2;

  if (gy >= rows || gx >= cols)
    return;

  T c0;
  T c1;
  T4 c2;
  T c3;
  T c4;
  T c5;
  T c6;
  T4 c7;
  T c8;
  T c9;
  T c10;
  T c11;
  T4 c12;
  T c13;
  T c14;
  T c15;
  T c16;
  T4 c17;
  T c18;
  T c19;
  T c20;
  T c21;
  T4 c22;
  T c23;
  T c24;

  int x_leftmost = mad24(max(gx - 2, 0), TSIZE, srcOffset);
  int x_left = mad24(max(gx - 1, 0), TSIZE, srcOffset);
  int x_central = mad24(gx, TSIZE, srcOffset);
  int x_right = mad24(min(gx + 4, cols - 1), TSIZE, srcOffset);
  int x_rightmost = mad24(min(gx + 5, cols - 1), TSIZE, srcOffset);

  int xdst = mad24(gx, TSIZE, dstOffset);

  // 0 line
  int src_index = max(gy - 2, 0) * srcStep;
  c0 = *(__global T *)(srcptr + src_index + x_leftmost);
  c1 = *(__global T *)(srcptr + src_index + x_left);
  LOAD4(c2, x_central);
  c3 = *(__global T *)(srcptr + src_index + x_right);
  c4 = *(__global T *)(srcptr + src_index + x_rightmost);

  // 1 line
  src_index = max(gy - 1, 0) * srcStep;
  c5 = *(__global T *)(srcptr + src_index + x_leftmost);
  c6 = *(__global T *)(srcptr + src_index + x_left);
  LOAD4(c7, x_central);
  c8 = *(__global T *)(srcptr + src_index + x_right);
  c9 = *(__global T *)(srcptr + src_index + x_rightmost);

  // 2 line
  src_index = gy * srcStep;
  c10 = *(__global T *)(srcptr + src_index + x_leftmost);
  c11 = *(__global T *)(srcptr + src_index + x_left);
  LOAD4(c12, x_central);
  c13 = *(__global T *)(srcptr + src_index + x_right);
  c14 = *(__global T *)(srcptr + src_index + x_rightmost);

  // 3 line
  src_index = (gy + 1) * srcStep;
  c15 = *(__global T *)(srcptr + src_index + x_leftmost);
  c16 = *(__global T *)(srcptr + src_index + x_left);
  LOAD4(c17, x_central);
  c18 = *(__global T *)(srcptr + src_index + x_right);
  c19 = *(__global T *)(srcptr + src_index + x_rightmost);

  for (int k = 0; k < 4; k++) {
    // 4 line
    src_index = min(gy + k + 2, rows - 1) * srcStep;
    c20 = *(__global T *)(srcptr + src_index + x_leftmost);
    c21 = *(__global T *)(srcptr + src_index + x_left);
    LOAD4(c22, x_central);
    c23 = *(__global T *)(srcptr + src_index + x_right);
    c24 = *(__global T *)(srcptr + src_index + x_rightmost);

    T4 p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15,
        p16, p17, p18, p19, p20, p21, p22, p23, p24;

    SHUFFLE4_5(c0, c1, c2, c3, c4, p0, p1, p2, p3, p4);

    SHUFFLE4_5(c5, c6, c7, c8, c9, p5, p6, p7, p8, p9);

    SHUFFLE4_5(c10, c11, c12, c13, c14, p10, p11, p12, p13, p14);

    SHUFFLE4_5(c15, c16, c17, c18, c19, p15, p16, p17, p18, p19);

    SHUFFLE4_5(c20, c21, c22, c23, c24, p20, p21, p22, p23, p24);

    T4 mid;

    OP(p1, p2);
    OP(p0, p1);
    OP(p1, p2);
    OP(p4, p5);
    OP(p3, p4);
    OP(p4, p5);
    OP(p0, p3);
    OP(p2, p5);
    OP(p2, p3);
    OP(p1, p4);
    OP(p1, p2);
    OP(p3, p4);
    OP(p7, p8);
    OP(p6, p7);
    OP(p7, p8);
    OP(p10, p11);
    OP(p9, p10);
    OP(p10, p11);
    OP(p6, p9);
    OP(p8, p11);
    OP(p8, p9);
    OP(p7, p10);
    OP(p7, p8);
    OP(p9, p10);
    OP(p0, p6);

    OP(p4, p10);
    OP(p4, p6);
    OP(p2, p8);
    OP(p2, p4);
    OP(p6, p8);
    OP(p1, p7);
    OP(p5, p11);
    OP(p5, p7);
    OP(p3, p9);
    OP(p3, p5);
    OP(p7, p9);
    OP(p1, p2);
    OP(p3, p4);
    OP(p5, p6);
    OP(p7, p8);
    OP(p9, p10);
    OP(p13, p14);
    OP(p12, p13);
    OP(p13, p14);
    OP(p16, p17);
    OP(p15, p16);
    OP(p16, p17);
    OP(p12, p15);
    OP(p14, p17);
    OP(p14, p15);

    OP(p13, p16);
    OP(p13, p14);
    OP(p15, p16);
    OP(p19, p20);
    OP(p18, p19);
    OP(p19, p20);
    OP(p21, p22);
    OP(p23, p24);
    OP(p21, p23);
    OP(p22, p24);
    OP(p22, p23);
    OP(p18, p21);
    OP(p20, p23);
    OP(p20, p21);
    OP(p19, p22);
    OP(p22, p24);
    OP(p19, p20);
    OP(p21, p22);
    OP(p23, p24);
    OP(p12, p18);
    OP(p16, p22);
    OP(p16, p18);
    OP(p14, p20);
    OP(p20, p24);
    OP(p14, p16);

    OP(p18, p20);
    OP(p22, p24);
    OP(p13, p19);
    OP(p17, p23);
    OP(p17, p19);
    OP(p15, p21);
    OP(p15, p17);
    OP(p19, p21);
    OP(p13, p14);
    OP(p15, p16);
    OP(p17, p18);
    OP(p19, p20);
    OP(p21, p22);
    OP(p23, p24);
    OP(p0, p12);
    OP(p8, p20);
    OP(p8, p12);
    OP(p4, p16);
    OP(p16, p24);
    OP(p12, p16);
    OP(p2, p14);
    OP(p10, p22);
    OP(p10, p14);
    OP(p6, p18);
    OP(p6, p10);
    OP(p10, p12);
    OP(p1, p13);
    OP(p9, p21);
    OP(p9, p13);
    OP(p5, p17);
    OP(p13, p17);
    OP(p3, p15);
    OP(p11, p23);
    OP(p11, p15);
    OP(p7, p19);
    OP(p7, p11);
    OP(p11, p13);
    OP(p11, p12);

    int dst_index = mad24(gy + k, dstStep, xdst);

    STORE4(p12, dst_index);

    c0 = c5;
    c1 = c6;
    c2 = c7;
    c3 = c8;
    c4 = c9;
    c5 = c10;
    c6 = c11;
    c7 = c12;
    c8 = c13;
    c9 = c14;
    c10 = c15;
    c11 = c16;
    c12 = c17;
    c13 = c18;
    c14 = c19;
    c15 = c20;
    c16 = c21;
    c17 = c22;
    c18 = c23;
    c19 = c24;
  }
}

#endif

__kernel void medianFilter3(__global const uchar *srcptr, int src_step,
                            int src_offset, __global uchar *dstptr,
                            int dst_step, int dst_offset, int dst_rows,
                            int dst_cols) {
  __local T data[18][18];

  int x = get_local_id(0);
  int y = get_local_id(1);

  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int dx = gx - x - 1;
  int dy = gy - y - 1;

  int id = min(mad24(x, 16, y), 9 * 18 - 1);

  int dr = id / 18;
  int dc = id % 18;

  int c = clamp(dx + dc, 0, dst_cols - 1);

  int r = clamp(dy + dr, 0, dst_rows - 1);
  int index1 = mad24(r, src_step, mad24(c, TSIZE, src_offset));
  r = clamp(dy + dr + 9, 0, dst_rows - 1);
  int index9 = mad24(r, src_step, mad24(c, TSIZE, src_offset));

  data[dr][dc] = loadpix(srcptr + index1);
  data[dr + 9][dc] = loadpix(srcptr + index9);
  barrier(CLK_LOCAL_MEM_FENCE);

  T p0 = data[y][x], p1 = data[y][(x + 1)], p2 = data[y][(x + 2)];
  T p3 = data[y + 1][x], p4 = data[y + 1][(x + 1)], p5 = data[y + 1][(x + 2)];
  T p6 = data[y + 2][x], p7 = data[y + 2][(x + 1)], p8 = data[y + 2][(x + 2)];
  T mid;

  OP(p1, p2);
  OP(p4, p5);
  OP(p7, p8);
  OP(p0, p1);
  OP(p3, p4);
  OP(p6, p7);
  OP(p1, p2);
  OP(p4, p5);
  OP(p7, p8);
  OP(p0, p3);
  OP(p5, p8);
  OP(p4, p7);
  OP(p3, p6);
  OP(p1, p4);
  OP(p2, p5);
  OP(p4, p7);
  OP(p4, p2);
  OP(p6, p4);
  OP(p4, p2);

  int dst_index = mad24(gy, dst_step, mad24(gx, TSIZE, dst_offset));

  if (gy < dst_rows && gx < dst_cols)
    storepix(p4, dstptr + dst_index);
}

__kernel void medianFilter5(__global const uchar *srcptr, int src_step,
                            int src_offset, __global uchar *dstptr,
                            int dst_step, int dst_offset, int dst_rows,
                            int dst_cols) {
  __local T data[20][20];

  int x = get_local_id(0);
  int y = get_local_id(1);

  int gx = get_global_id(0);
  int gy = get_global_id(1);

  int dx = gx - x - 2;
  int dy = gy - y - 2;

  int id = min(mad24(x, 16, y), 10 * 20 - 1);

  int dr = id / 20;
  int dc = id % 20;

  int c = clamp(dx + dc, 0, dst_cols - 1);
  int r = clamp(dy + dr, 0, dst_rows - 1);
  int index1 = mad24(r, src_step, mad24(c, TSIZE, src_offset));

  r = clamp(dy + dr + 10, 0, dst_rows - 1);
  int index10 = mad24(r, src_step, mad24(c, TSIZE, src_offset));

  data[dr][dc] = loadpix(srcptr + index1);
  data[dr + 10][dc] = loadpix(srcptr + index10);
  barrier(CLK_LOCAL_MEM_FENCE);

  T p0 = data[y][x], p1 = data[y][x + 1], p2 = data[y][x + 2],
    p3 = data[y][x + 3], p4 = data[y][x + 4];
  T p5 = data[y + 1][x], p6 = data[y + 1][x + 1], p7 = data[y + 1][x + 2],
    p8 = data[y + 1][x + 3], p9 = data[y + 1][x + 4];
  T p10 = data[y + 2][x], p11 = data[y + 2][x + 1], p12 = data[y + 2][x + 2],
    p13 = data[y + 2][x + 3], p14 = data[y + 2][x + 4];
  T p15 = data[y + 3][x], p16 = data[y + 3][x + 1], p17 = data[y + 3][x + 2],
    p18 = data[y + 3][x + 3], p19 = data[y + 3][x + 4];
  T p20 = data[y + 4][x], p21 = data[y + 4][x + 1], p22 = data[y + 4][x + 2],
    p23 = data[y + 4][x + 3], p24 = data[y + 4][x + 4];
  T mid;

  OP(p1, p2);
  OP(p0, p1);
  OP(p1, p2);
  OP(p4, p5);
  OP(p3, p4);
  OP(p4, p5);
  OP(p0, p3);
  OP(p2, p5);
  OP(p2, p3);
  OP(p1, p4);
  OP(p1, p2);
  OP(p3, p4);
  OP(p7, p8);
  OP(p6, p7);
  OP(p7, p8);
  OP(p10, p11);
  OP(p9, p10);
  OP(p10, p11);
  OP(p6, p9);
  OP(p8, p11);
  OP(p8, p9);
  OP(p7, p10);
  OP(p7, p8);
  OP(p9, p10);
  OP(p0, p6);
  OP(p4, p10);
  OP(p4, p6);
  OP(p2, p8);
  OP(p2, p4);
  OP(p6, p8);
  OP(p1, p7);
  OP(p5, p11);
  OP(p5, p7);
  OP(p3, p9);
  OP(p3, p5);
  OP(p7, p9);
  OP(p1, p2);
  OP(p3, p4);
  OP(p5, p6);
  OP(p7, p8);
  OP(p9, p10);
  OP(p13, p14);
  OP(p12, p13);
  OP(p13, p14);
  OP(p16, p17);
  OP(p15, p16);
  OP(p16, p17);
  OP(p12, p15);
  OP(p14, p17);
  OP(p14, p15);
  OP(p13, p16);
  OP(p13, p14);
  OP(p15, p16);
  OP(p19, p20);
  OP(p18, p19);
  OP(p19, p20);
  OP(p21, p22);
  OP(p23, p24);
  OP(p21, p23);
  OP(p22, p24);
  OP(p22, p23);
  OP(p18, p21);
  OP(p20, p23);
  OP(p20, p21);
  OP(p19, p22);
  OP(p22, p24);
  OP(p19, p20);
  OP(p21, p22);
  OP(p23, p24);
  OP(p12, p18);
  OP(p16, p22);
  OP(p16, p18);
  OP(p14, p20);
  OP(p20, p24);
  OP(p14, p16);
  OP(p18, p20);
  OP(p22, p24);
  OP(p13, p19);
  OP(p17, p23);
  OP(p17, p19);
  OP(p15, p21);
  OP(p15, p17);
  OP(p19, p21);
  OP(p13, p14);
  OP(p15, p16);
  OP(p17, p18);
  OP(p19, p20);
  OP(p21, p22);
  OP(p23, p24);
  OP(p0, p12);
  OP(p8, p20);
  OP(p8, p12);
  OP(p4, p16);
  OP(p16, p24);
  OP(p12, p16);
  OP(p2, p14);
  OP(p10, p22);
  OP(p10, p14);
  OP(p6, p18);
  OP(p6, p10);
  OP(p10, p12);
  OP(p1, p13);
  OP(p9, p21);
  OP(p9, p13);
  OP(p5, p17);
  OP(p13, p17);
  OP(p3, p15);
  OP(p11, p23);
  OP(p11, p15);
  OP(p7, p19);
  OP(p7, p11);
  OP(p11, p13);
  OP(p11, p12);

  int dst_index = mad24(gy, dst_step, mad24(gx, TSIZE, dst_offset));

  if (gy < dst_rows && gx < dst_cols)
    storepix(p12, dstptr + dst_index);
}

__kernel void medianFilter7(__global const uchar *srcptr, int src_step,
                            int src_offset, __global uchar *dstptr,
                            int dst_step, int dst_offset, int dst_rows,
                            int dst_cols) {
  // Increase local buffer size and add padding
  __local T data[24][24];

  // Calculate global and local indices
  int x = get_local_id(0);
  int y = get_local_id(1);
  int gx = get_global_id(0);
  int gy = get_global_id(1);

  // Calculate offset for accessing global memory
  int dx = gx - x - 3;
  int dy = gy - y - 3;

  // Calculate flattened index for local buffer
  int id = min(mad24(x, 18, y), 11 * 24 - 1);
  int dr = id / 24;
  int dc = id % 24;

  // Clamp indices to image boundaries
  int c = clamp(dx + dc, 0, dst_cols - 1);
  int r = clamp(dy + dr, 0, dst_rows - 1);
  int index1 = mad24(r, src_step, mad24(c, TSIZE, src_offset));

  // Adjust for padding
  r = clamp(dy + dr + 11, 0, dst_rows - 1);
  int index11 = mad24(r, src_step, mad24(c, TSIZE, src_offset));

  // Load data into local buffer
  data[dr][dc] = loadpix(srcptr + index1);
  data[dr + 11][dc] = loadpix(srcptr + index11);
  barrier(CLK_LOCAL_MEM_FENCE);

  T p[49];

  // Copy data from local buffer to array
#pragma unroll
  for (int i = 0; i < 7; ++i) {
#pragma unroll
    for (int j = 0; j < 7; ++j) {
      p[i * 7 + j] = data[y + i][x + j];
    }
  }

  // Perform bubble sort on the array
#pragma unroll
  for (int i = 0; i < 48; ++i) {
    for (int j = 0; j < 48 - i; ++j) {
      T1 a = *((uchar *)(&p[j]));
      T1 b = *((uchar *)(&p[j + 1]));
      if (a > b) {
        T temp = p[j];
        p[j] = p[j + 1];
        p[j + 1] = temp;
      }
    }
  }

  // Select median value
  T median = p[24];

  // Calculate destination index
  int dst_index = mad24(gy, dst_step, mad24(gx, TSIZE, dst_offset));

  // Store median value to destination buffer
  if (gy < dst_rows && gx < dst_cols)
    storepix(median, dstptr + dst_index);
}

// https://github.com/aglenis/gpu_medfilter/blob/a29aaea52a2bd69f24fd4071ec3d5380eacf9f74/opencl_implementation/kernel_wrapper.cpp

#ifndef WINDOW_SIZE
#define WINDOW_SIZE 5
#endif

void bubbleSort(unsigned char v[], int size) {
  // bubble-sort
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (v[i] > v[j]) { /* swap? */
        unsigned char tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
      }
    }
  }
}

// __constant int WINDOW_SIZE=(int)sqrt((float)ARRAY_SIZE_ARG);

#ifndef filter_offset
#define filter_offset (WINDOW_SIZE / 2)
#endif

#define ARRAY_SIZE_ARG (WINDOW_SIZE * WINDOW_SIZE)

__kernel void MedianFilter2D(__global const uchar *input,
                             __global uchar *output, int widthImage,
                             int heightImage) {
  unsigned int y = get_global_id(0);
  unsigned int x = get_global_id(1);

  if (y >= heightImage || x >= widthImage)
    return;

  unsigned char window[ARRAY_SIZE_ARG];
  int count = 0;

#pragma unroll
  for (int k = -filter_offset; k <= filter_offset; k++) {
#pragma unroll
    for (int l = -filter_offset; l <= filter_offset; l++) {
      int imageY = y + k;
      int imageX = x + l;

      // Check boundary conditions
      if (imageY >= 0 && imageY < heightImage && imageX >= 0 &&
          imageX < widthImage)
        window[count++] = input[imageY * widthImage + imageX];
    }
  }

  bubbleSort(window, count);
  output[y * widthImage + x] = window[count / 2];
}

void bubbleSortShared(__local char *v, int size) {
  // bubble-sort
  for (int i = 0; i < size; i++) {
    for (int j = i + 1; j < size; j++) {
      if (v[i] > v[j]) { /* swap? */
        unsigned char tmp = v[i];
        v[i] = v[j];
        v[j] = tmp;
      }
    }
  }
}

unsigned char medianBubble(unsigned char *dataQueue, int lengthQueue) {
  int minValueIndex;
  unsigned char bufferData;
  int i, j;

  for (j = 0; j <= (lengthQueue - 1) / 2; j++) {
    minValueIndex = j;
    for (i = j + 1; i < lengthQueue; i++)
      if (dataQueue[i] < dataQueue[minValueIndex])
        minValueIndex = i;

    bufferData = dataQueue[j];
    dataQueue[j] = dataQueue[minValueIndex];
    dataQueue[minValueIndex] = bufferData;
  }

  return dataQueue[(lengthQueue - 1) / 2];
}

#define TILE_SIZE 16 // block size for each dimension
#define CHANNELS 3   // assuming RGB channels

#ifndef KERNEL_SIZE
#define KERNEL_SIZE 5
#endif

__kernel void MedianFilter3D(__global const unsigned char *input,
                             __global unsigned char *output, const int width,
                             const int height) {
  const int grayWidthStep = width;
  const int colorWidthStep = 3 * width;

  int halfWindow = (int)KERNEL_SIZE / 2;
  const int col_index = get_global_id(0);
  const int row_index = get_global_id(1);

  __local unsigned char
      sharedmem[CHANNELS][TILE_SIZE + KERNEL_SIZE - 1]
               [TILE_SIZE + KERNEL_SIZE - 1]; // initialize shared memory
  bool is_x_left = (get_local_id(0) == 0),
       is_x_right = (get_local_id(0) == TILE_SIZE - 1);
  bool is_y_top = (get_local_id(1) == 0),
       is_y_bottom = (get_local_id(1) == TILE_SIZE - 1);

  if (row_index < height && col_index < width) {
    // Location of pixel in output
    const int median_pixel_index = (row_index * width + col_index) * CHANNELS;
    for (int chan = 0; chan < CHANNELS; chan++)
      sharedmem[chan][get_local_id(0) + halfWindow][get_local_id(1) +
                                                    halfWindow] =
          input[row_index * colorWidthStep + col_index * CHANNELS + chan];

    if (is_x_left && (col_index > halfWindow - 1))
      for (int j = 1; j <= halfWindow; j++)
        for (int chan = 0; chan < CHANNELS; chan++)
          sharedmem[chan][get_local_id(0) + halfWindow - j]
                   [get_local_id(1) + halfWindow] =
                       input[row_index * colorWidthStep +
                             (col_index - j) * CHANNELS + chan];
    else if (is_x_right && (col_index < width - halfWindow))
      for (int j = 1; j <= halfWindow; j++)
        for (int chan = 0; chan < CHANNELS; chan++)
          sharedmem[chan][get_local_id(0) + halfWindow + j]
                   [get_local_id(1) + halfWindow] =
                       input[row_index * colorWidthStep +
                             (col_index + j) * CHANNELS + chan];
    if (is_y_top && (row_index > halfWindow - 1)) {
      for (int j = 1; j <= halfWindow; j++)
        for (int chan = 0; chan < CHANNELS; chan++)
          sharedmem[chan][get_local_id(0) + halfWindow]
                   [get_local_id(1) + halfWindow - j] =
                       input[(row_index - j) * colorWidthStep +
                             col_index * CHANNELS + chan];
      if (is_x_left)
        for (int j = 1; j <= halfWindow; j++)
          for (int i = 1; i <= halfWindow; i++)
            for (int chan = 0; chan < CHANNELS; chan++)
              sharedmem[chan][get_local_id(0) + halfWindow - i]
                       [get_local_id(1) + halfWindow - j] =
                           input[(row_index - j) * colorWidthStep +
                                 (col_index - i) * CHANNELS + chan];
      else if (is_x_right)
        for (int j = 1; j <= halfWindow; j++)
          for (int i = 1; i <= halfWindow; i++)
            for (int chan = 0; chan < CHANNELS; chan++)
              sharedmem[chan][get_local_id(0) + halfWindow + i]
                       [get_local_id(1) + halfWindow - j] =
                           input[(row_index - j) * colorWidthStep +
                                 (col_index + i) * CHANNELS + chan];
    } else if (is_y_bottom && (row_index < height - halfWindow)) {
      for (int j = 1; j <= halfWindow; j++)
        for (int chan = 0; chan < CHANNELS; chan++)
          sharedmem[chan][get_local_id(0) + halfWindow]
                   [get_local_id(1) + halfWindow + j] =
                       input[(row_index + j) * colorWidthStep +
                             col_index * CHANNELS + chan];
      if (is_x_right)
        for (int j = 1; j <= halfWindow; j++)
          for (int i = 1; i <= halfWindow; i++)
            for (int chan = 0; chan < CHANNELS; chan++)
              sharedmem[chan][get_local_id(0) + halfWindow + i]
                       [get_local_id(1) + halfWindow + j] =
                           input[(row_index + j) * colorWidthStep +
                                 (col_index + i) * CHANNELS + chan];
      else if (is_x_left)
        for (int j = 1; j <= halfWindow; j++)
          for (int i = 1; i <= halfWindow; i++)
            for (int chan = 0; chan < CHANNELS; chan++)
              sharedmem[chan][get_local_id(0) + halfWindow - i]
                       [get_local_id(1) + halfWindow + j] =
                           input[(row_index + j) * colorWidthStep +
                                 (col_index - i) * CHANNELS + chan];
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    if ((row_index >= KERNEL_SIZE / 2) &&
        (row_index < height - KERNEL_SIZE / 2) &&
        (col_index >= KERNEL_SIZE / 2) &&
        (col_index < width - KERNEL_SIZE / 2)) {
      unsigned char median_array[KERNEL_SIZE * KERNEL_SIZE * CHANNELS];
      int median_array_index = 0;
      for (int chan = 0; chan < CHANNELS; chan++) {
#pragma unroll
        for (int i = 0; i < KERNEL_SIZE; i++) {
#pragma unroll
          for (int j = 0; j < KERNEL_SIZE; j++) {
            if (median_array_index < KERNEL_SIZE * KERNEL_SIZE * CHANNELS)
              median_array[median_array_index] =
                  sharedmem[chan][get_local_id(0) + i][get_local_id(1) + j];
            median_array_index++;
          }
        }
      }
      // Ignore pixels which require padding for its window (example:corner and
      // boundary pixels). Those pixels are initialized to 0
      if (row_index < height && col_index < width) {
        // Apply median filter to each channel separately
        for (int chan = 0; chan < CHANNELS; chan++) {
          unsigned char median_array[KERNEL_SIZE * KERNEL_SIZE];
          int median_array_index = 0;

// Fill the median array
#pragma unroll
          for (int i = 0; i < KERNEL_SIZE; i++) {
#pragma unroll
            for (int j = 0; j < KERNEL_SIZE; j++) {
              median_array[median_array_index] =
                  sharedmem[chan][get_local_id(0) + i][get_local_id(1) + j];
              median_array_index++;
            }
          }

          // Compute median value
          output[(row_index * width + col_index) * CHANNELS + chan] =
              medianBubble(median_array, KERNEL_SIZE * KERNEL_SIZE);
        }
      }
    }
  }
}

/*
__kernel void MedianFilter3DNoShared(__global const uchar *input,
                                     __global uchar *output, int widthImage,
                                     int heightImage) {
  unsigned int y = get_global_id(0);
  unsigned int x = get_global_id(1);

  if (y >= heightImage || x >= widthImage)
    return;

  unsigned char windowR[KERNEL_SIZE * KERNEL_SIZE];
  unsigned char windowG[KERNEL_SIZE * KERNEL_SIZE];
  unsigned char windowB[KERNEL_SIZE * KERNEL_SIZE];

  int countR = 0;
  int countG = 0;
  int countB = 0;

  for (int j = -KERNEL_SIZE / 2; j <= KERNEL_SIZE / 2; j++) {
    for (int i = -KERNEL_SIZE / 2; i <= KERNEL_SIZE / 2; i++) {
      int imageY = y + j;
      int imageX = x + i;

      // Check boundary conditions
      if (imageY >= 0 && imageY < heightImage && imageX >= 0 &&
          imageX < widthImage) {

        int inputIndex = (imageY * widthImage + imageX) * 3;

        windowR[countR++] = input[inputIndex];

        windowG[countG++] = input[inputIndex + 1];

        windowB[countB++] = input[inputIndex + 2];
      }
    }
  }

  // Sort each channel separately

  bubbleSort(windowR, countR);
  bubbleSort(windowG, countG);
  bubbleSort(windowB, countB);

  // Write the median value to the output
  int outputIndex = (y * widthImage + x) * 3;

  output[outputIndex] = windowR[countR / 2];
  output[outputIndex + 1] = windowG[countG / 2];
  output[outputIndex + 2] = windowB[countB / 2];
}
*/