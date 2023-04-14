
import std.stdio;
import std.process;
import std.datetime.stopwatch : StopWatch;
import std.math : round;

import dcv.core;
import dcv.imageio;
import dcv.plot.figure;

import mir.ndslice, mir.rc;

import oclcv;

void main()
{
    CLContext context = new CLContext;
    context.clInfo().writeln;
    
    auto imRGBI = imread("lena.png");
    scope(exit) destroyFree(imRGBI);

    auto imRGB = imRGBI.sliced;

    auto srcH = imRGB.shape[0];
    auto srcW = imRGB.shape[1];

    auto resizer = new Resize3!"resize_bicubic"(cast(int)srcH, cast(int)srcW, cast(int)(srcH/2), cast(int)(srcW/2), context);
    auto d_rgb = new CLBuffer(context, BufferMeta(UBYTE, srcH, srcW, 3));

    d_rgb.upload(imRGB.ptr[0..imRGB.elementCount]);

    auto d_resizedRGB = resizer.run(d_rgb);
    
    auto imResized = rcslice!ubyte([cast(size_t)(srcH/2), cast(size_t)(srcW/2), 3], 0);

    d_resizedRGB.download(imResized.ptr[0..imResized.elementCount]);

    imshow(imResized, "resized");

    waitKey();
    
    destroyFigures();
}