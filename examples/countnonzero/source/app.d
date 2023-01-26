import std.stdio;

import core.stdc.stdio;
import core.stdc.stdlib;
import core.stdc.time;

import dcv.plot, dcv.imageio, dcv.imgproc, dcv.core;

import mir.ndslice;

import oclcv;

void main()
{
    CLContext context = new CLContext;
    context.clInfo().writeln;

    auto img = imread("data/test.png").sliced.rgb2gray;


    size_t height = img.shape[0];
    size_t width = img.shape[1];

    auto counterGT = new CountNonZero(cast(int)height, cast(int)width, ">", context); // "Greater than" search
    
    auto d_img = new CLBuffer(context, BufferMeta(UBYTE, height, width, 1));
    
    d_img.upload(img.ptr[0..img.elementCount]);
    
    
    size_t count = counterGT.run(d_img, 0); // check if values > 0

    writeln("number of nonzero elements: ", count);
}
