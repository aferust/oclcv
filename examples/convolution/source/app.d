
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

    auto imGrayFloat = imRGBI.sliced.rgb2gray.as!float.rcslice;

    auto srcH = imGrayFloat.shape[0];
    auto srcW = imGrayFloat.shape[1];

    auto gauss5x5filter = ([1.0f, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 36,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1].staticArray[].sliced(5,5) * 1/256.0f).rcslice;
    
    auto d_filter = mallocNew!CLBuffer(context, BufferMeta(FLOAT, cast(int)gauss5x5filter.shape[0], cast(int)gauss5x5filter.shape[1], 1));
    scope(exit) destroyFree(d_filter);
    d_filter.upload(gauss5x5filter.ptr[0..gauss5x5filter.elementCount]);

    auto conv = mallocNew!Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gauss5x5filter.shape[0], cast(int)gauss5x5filter.shape[1], context);
    
    auto d_input = mallocNew!CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    scope(exit) destroyFree(d_input);
    d_input.upload(imGrayFloat.ptr[0..imGrayFloat.elementCount]);

    auto d_filtered = conv.run(d_input, d_filter);
    scope(exit) destroyFree(d_filtered);
    
    auto imFiltered = uninitRCslice!float(imGrayFloat.shape);

    d_filtered.download(imFiltered.ptr[0..imFiltered.elementCount]);

    // filter color image
    // not a best implementation but this is still ~10 times faster then conv of DCV;
    StopWatch s;
    s.start;

    auto imRGB = imRGBI.sliced.as!float.rcslice;
    auto d_inputR = mallocNew!CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    auto d_inputG = mallocNew!CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    auto d_inputB = mallocNew!CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    d_inputR.upload(imRGB[0..$, 0..$, 0].rcslice.ptr[0..srcH*srcW]);
    d_inputG.upload(imRGB[0..$, 0..$, 1].rcslice.ptr[0..srcH*srcW]);
    d_inputB.upload(imRGB[0..$, 0..$, 2].rcslice.ptr[0..srcH*srcW]);

    scope(exit){
        destroyFree(d_inputR);
        destroyFree(d_inputG);
        destroyFree(d_inputB);
    }

    auto gaussianKernel = gaussian!float(2, 5, 5);

    auto d_filter_u = mallocNew!CLBuffer(context, BufferMeta(FLOAT, cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1]));
    scope(exit) destroyFree(d_filter_u);
    d_filter_u.upload(gaussianKernel.ptr[0..gaussianKernel.elementCount]);

    auto convR = mallocNew!Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1], context);
    auto convG = mallocNew!Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1], context);
    auto convB = mallocNew!Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1], context);
    
    scope(exit){
        destroyFree(convR);
        destroyFree(convG);
        destroyFree(convB);
    }

    auto d_filteredR = convR.run(d_inputR, d_filter_u);
    auto d_filteredG = convG.run(d_inputG, d_filter_u);
    auto d_filteredB = convB.run(d_inputB, d_filter_u);

    scope(exit){
        destroyFree(d_filteredR);
        destroyFree(d_filteredG);
        destroyFree(d_filteredB);
    }

    auto imFilteredR = uninitRCslice!float(srcH, srcW);
    auto imFilteredG = uninitRCslice!float(srcH, srcW);
    auto imFilteredB = uninitRCslice!float(srcH, srcW);
    d_filteredR.download(imFilteredR.ptr[0..srcH*srcW]);
    d_filteredG.download(imFilteredG.ptr[0..srcH*srcW]);
    d_filteredB.download(imFilteredB.ptr[0..srcH*srcW]);

    auto merged  = uninitRCslice!float(srcH, srcW, 3);
    merged[0..$, 0..$, 0] = imFilteredR;
    merged[0..$, 0..$, 1] = imFilteredG;
    merged[0..$, 0..$, 2] = imFilteredB;
    
    s.stop();
    printf("%d msecs\n", s.peek.total!"msecs");

    imshow(imRGBI, "original");
    imshow(imGrayFloat, "imGrayFloat");
    imshow(imFiltered, "imFiltered");
    imshow(merged, "imFiltered 3");

    waitKey();
    
    destroyFigures();
}