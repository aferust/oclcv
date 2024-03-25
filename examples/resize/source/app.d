
import std.stdio;
import std.process;
import std.datetime.stopwatch : StopWatch;
import std.math : round;

import dcv.core;
import dcv.imageio;
import dcv.plot.figure;

import mir.ndslice, mir.rc;

import oclcv;

@nogc nothrow:

void main()
{
    CLContext context = mallocNew!CLContext;
    scope(exit) destroyFree(context);
    
    auto imRGBI = imread("lena.png");
    scope(exit) destroyFree(imRGBI);

    auto imRGB = imRGBI.sliced;

    auto srcH = imRGB.shape[0];
    auto srcW = imRGB.shape[1];

    auto resizer = mallocNew!(Resize3!"resize_bicubic")(cast(int)srcH, cast(int)srcW, cast(int)(srcH/2), cast(int)(srcW/2), context);
    scope(exit) destroyFree(resizer);
    auto d_rgb = mallocNew!CLBuffer(context, BufferMeta(UBYTE, srcH, srcW, 3));
    scope(exit) destroyFree(d_rgb);

    d_rgb.upload(imRGB.ptr[0..imRGB.elementCount]);

    auto d_resizedRGB = resizer.run(d_rgb);
    scope(exit) destroyFree(d_resizedRGB);
    
    auto imResized = rcslice!ubyte([cast(size_t)(srcH/2), cast(size_t)(srcW/2), 3], 0);

    d_resizedRGB.download(imResized.ptr[0..imResized.elementCount]);

    imshow(imResized, "resized");

    waitKey();
    
    destroyFigures();
}