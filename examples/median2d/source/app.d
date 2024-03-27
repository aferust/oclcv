
import core.stdc.stdio;

import dcv.core, dcv.imgproc;
import dcv.imageio;
import dcv.plot.figure;
import std.datetime.stopwatch : StopWatch;
import std.array : staticArray;

import mir.ndslice, mir.rc;

import oclcv;

@nogc nothrow:

void main()
{
    CLContext context = mallocNew!CLContext;
    scope(exit) destroyFree(context);

    printf("%s\n", context.clInfo().data.ptr);
    
    auto imRGBI = imread("lena.png");
    scope(exit) destroyFree(imRGBI);
    
    auto imGray = imRGBI.sliced.rgb2gray;

    auto srcH = imGray.shape[0];
    auto srcW = imGray.shape[1];
    
    auto median2d = mallocNew!Median2D(cast(int)srcH, cast(int)srcW, 5, context);
    
    auto d_input = mallocNew!CLBuffer(context, BufferMeta(UBYTE, srcH, srcW, 1));
    scope(exit) destroyFree(d_input);
    d_input.upload(imGray.ptr[0..imGray.elementCount]);

    auto d_filtered = median2d.run(d_input);
    scope(exit) destroyFree(d_filtered);
    
    auto imFiltered = uninitRCslice!ubyte(imGray.shape);

    d_filtered.download(imFiltered.ptr[0..imFiltered.elementCount]);

    imshow(imRGBI, "original");
    imshow(imGray, "imGray");
    imshow(imFiltered, "2D imFiltered median");

    auto rgbSlice = imRGBI.sliced;
    auto imFiltered3c = uninitRCslice!ubyte(rgbSlice.shape);

    auto median3d = mallocNew!Median3D(cast(int)srcH, cast(int)srcW, 5, context); scope(exit) destroyFree(median3d);
    auto d_input3c = mallocNew!CLBuffer(context, BufferMeta(UBYTE, srcH, srcW, 3));
    scope(exit) destroyFree(d_input3c);
    
    StopWatch s;
    s.start;

    d_input3c.upload(rgbSlice.ptr[0..rgbSlice.elementCount]);
    auto d_filtered3c = median3d.run(d_input3c);
    scope(exit) destroyFree(d_filtered3c);
    
    s.stop();
    
    d_filtered3c.download(imFiltered3c.ptr[0..imFiltered3c.elementCount]);

    
    printf("%llu msecs\n", s.peek.total!"msecs");

    imshow(rgbSlice, "original");
    imshow(imFiltered3c, "median 3c");

    waitKey();
    
    destroyFigures();
}