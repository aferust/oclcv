import std.stdio;

import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.time;

import dcv.plot, dcv.imageio, dcv.imgproc, dcv.core;

import mir.ndslice;

import bindbc.opencl;

import oclcv;

void main()
{
    auto _ret = loadOpenCL();
    writeln("Load CL: ", _ret);

    CLContext context = new CLContext;
    context.clInfo().writeln;

    auto leftrgb = imread("data/im0.png").sliced;
    auto rightrgb = imread("data/im1.png").sliced;

    size_t height = leftrgb.shape[0];
    size_t width = leftrgb.shape[1];

    auto convl = new RGB2GRAY(cast(int)height, cast(int)width, context);
    auto convr = new RGB2GRAY( cast(int)height, cast(int)width, context);
    
    auto d_left_rgb = new CLBuffer(context, BufferMeta(UBYTE, height, width, 3));
    auto d_right_rgb = new CLBuffer(context, BufferMeta(UBYTE, height, width, 3));
    d_left_rgb.upload(leftrgb.ptr);
    d_right_rgb.upload(rightrgb.ptr);
    
    CLBuffer d_left_gray = convl.run(d_left_rgb);
    CLBuffer d_right_gray = convr.run(d_right_rgb);

    auto disp = slice!short([height, width], 0);
    int disp_size = 128;
    StereoSGMCL ssgm = new StereoSGMCL(cast(int)height, cast(int)width, disp_size, context);

    clock_t begin = clock();
    
        auto d_dispmap = ssgm.run(d_left_gray, d_right_gray);
        d_dispmap.download(disp.ptr);
    
    clock_t end = clock(); printf("Elapsed time: %f \n", cast(double)(end - begin) / CLOCKS_PER_SEC);

    disp[] *= 10;
    imshow(disp, "disp");

    waitKey();
}
