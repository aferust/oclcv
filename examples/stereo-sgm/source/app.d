import std.stdio;

import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.time;

import dcv.plot, dcv.imageio, dcv.imgproc, dcv.core;

import mir.ndslice;

import oclcv;

@nogc nothrow:

void main()
{
    CLContext context = mallocNew!CLContext;
    scope(exit) destroyFree(context);

    auto _leftrgb = imread("data/im0.png");
    auto _rightrgb = imread("data/im1.png");

    scope(exit){
        destroyFree(_leftrgb);
        destroyFree(_rightrgb);
    }

    auto leftrgb = _leftrgb.sliced;
    auto rightrgb = _rightrgb.sliced;

    size_t height = leftrgb.shape[0];
    size_t width = leftrgb.shape[1];

    auto convl = mallocNew!RGB2GRAY(cast(int)height, cast(int)width, context);
    auto convr = mallocNew!RGB2GRAY( cast(int)height, cast(int)width, context);

    scope(exit){
        destroyFree(convl);
        destroyFree(convr);
    }
    
    auto d_left_rgb = mallocNew!CLBuffer(context, BufferMeta(UBYTE, height, width, 3));
    auto d_right_rgb = mallocNew!CLBuffer(context, BufferMeta(UBYTE, height, width, 3));
    scope(exit){
        destroyFree(d_left_rgb);
        destroyFree(d_right_rgb);
    }

    d_left_rgb.upload(leftrgb.ptr[0..leftrgb.elementCount]);
    d_right_rgb.upload(rightrgb.ptr[0..rightrgb.elementCount]);
    
    CLBuffer d_left_gray = convl.run(d_left_rgb);
    CLBuffer d_right_gray = convr.run(d_right_rgb);

    scope(exit){
        destroyFree(d_left_gray);
        destroyFree(d_right_gray);
    }

    auto disp = rcslice!short([height, width], 0);
    int disp_size = 128;
    StereoSGMCL ssgm = mallocNew!StereoSGMCL(cast(int)height, cast(int)width, disp_size, context);
    scope(exit) destroyFree(ssgm);

    clock_t begin = clock();
    
    auto d_dispmap = ssgm.run(d_left_gray, d_right_gray);
    scope(exit) destroyFree(d_dispmap);
    d_dispmap.download(disp.ptr[0..disp.elementCount]);
    
    clock_t end = clock(); printf("Elapsed time: %f \n", cast(double)(end - begin) / CLOCKS_PER_SEC);

    disp[] *= 10;
    imshow(disp, "disp");

    waitKey();

    destroyFigures();
}
