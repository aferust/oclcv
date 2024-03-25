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

    auto imgI = imread("data/test.png");
    scope(exit) destroyFree(imgI);

    auto img = imgI.sliced.rgb2gray;

    size_t height = img.shape[0];
    size_t width = img.shape[1];

    auto counterGT = mallocNew!CountNonZero(cast(int)height, cast(int)width, ">", context); // "Greater than" search
    scope(exit) destroyFree(counterGT);
    auto d_img = mallocNew!CLBuffer(context, BufferMeta(UBYTE, height, width, 1));
    scope(exit) destroyFree(d_img);
    
    d_img.upload(img.ptr[0..img.elementCount]);
    
    size_t count = counterGT.run(d_img, 0); // check if values > 0

    printf("number of nonzero elements: %llu", count);
}
