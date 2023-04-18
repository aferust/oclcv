
import std.stdio;

import dcv.core, dcv.imgproc;
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

    auto imGray = imRGBI.sliced.rgb2gray!ubyte;
    auto imGrayFloat = threshold!ubyte(imGray.lightScope, ubyte(128)).as!float.rcslice;

    auto srcH = imGrayFloat.shape[0];
    auto srcW = imGrayFloat.shape[1];

    auto filter = boxKernel!float(5, 5, 1.0f);
    
    auto d_filter = new CLBuffer(context, BufferMeta(FLOAT, cast(int)filter.shape[0], cast(int)filter.shape[1], 1));
    d_filter.upload(filter.ptr[0..filter.elementCount]);

    auto conv = new Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)filter.shape[0], cast(int)filter.shape[1], context);
    
    auto d_input = new CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    d_input.upload(imGrayFloat.ptr[0..imGrayFloat.elementCount]);

    auto d_filtered = conv.run(d_input, d_filter);
    
    auto imFiltered = uninitRCslice!float(imGrayFloat.shape);

    d_filtered.download(imFiltered.ptr[0..imFiltered.elementCount]);


    // 3d filter
    auto imRGB = imRGBI.sliced.as!float.rcslice;
    auto d_input3 = new CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 3));
    d_input3.upload(imRGB.ptr[0..imRGB.elementCount]);

    /*float[27] _kernelArr = [
        0,0,0,0,1,0,0,0,0,
        0,1,0,1,-6,1,0,1,0,
        0,0,0,0,1,0,0,0,0
    ];
    */
    float[27] _kernelArr = [
        2,3,2,3,6,3,2,3,2,
        3,6,3,6,-88,6,3,6,3,
        2,3,2,3,6,3,2,3,2
    ];
    auto filter3 = (_kernelArr[].sliced(3,3,3) * 1.0f/26.0f).rcslice;

    auto d_filter3 = new CLBuffer(context, BufferMeta(FLOAT, cast(int)filter3.shape[0], cast(int)filter3.shape[1], 3));
    d_filter3.upload(filter3.ptr[0..filter3.elementCount]);

    auto conv3 = new Convolution(cast(int)srcH, cast(int)srcW, 3,
        cast(int)filter3.shape[0], cast(int)filter3.shape[1], context);
    
    auto d_filtered3 = conv3.run(d_input3, d_filter3);
    auto imFiltered3 = uninitRCslice!float(imRGB.shape);
    d_filtered3.download(imFiltered3.ptr[0..imFiltered3.elementCount]);

    imshow(imRGBI, "original");
    imshow(imGrayFloat, "imGrayFloat");
    imshow(imFiltered, "imFiltered");
    imshow(imFiltered3, "imFiltered3");

    waitKey();
    
    destroyFigures();
}