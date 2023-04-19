
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

    auto imGrayFloat = imRGBI.sliced.rgb2gray.as!float.rcslice;

    auto srcH = imGrayFloat.shape[0];
    auto srcW = imGrayFloat.shape[1];

    auto gauss5x5filter = ([1.0f, 4, 6, 4, 1,
                    4, 16, 24, 16, 4,
                    6, 24, 36, 24, 36,
                    4, 16, 24, 16, 4,
                    1, 4, 6, 4, 1].sliced(5,5) * 1/256.0f).rcslice;
    
    auto d_filter = new CLBuffer(context, BufferMeta(FLOAT, cast(int)gauss5x5filter.shape[0], cast(int)gauss5x5filter.shape[1], 1));
    d_filter.upload(gauss5x5filter.ptr[0..gauss5x5filter.elementCount]);

    auto conv = new Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gauss5x5filter.shape[0], cast(int)gauss5x5filter.shape[1], context);
    
    auto d_input = new CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    d_input.upload(imGrayFloat.ptr[0..imGrayFloat.elementCount]);

    auto d_filtered = conv.run(d_input, d_filter);
    
    auto imFiltered = uninitRCslice!float(imGrayFloat.shape);

    d_filtered.download(imFiltered.ptr[0..imFiltered.elementCount]);

    // filter color image
    auto imRGB = imRGBI.sliced.as!float.rcslice;
    auto d_inputR = new CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    auto d_inputG = new CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    auto d_inputB = new CLBuffer(context, BufferMeta(FLOAT, srcH, srcW, 1));
    d_inputR.upload(imRGB[0..$, 0..$, 0].rcslice.ptr[0..srcH*srcW]);
    d_inputG.upload(imRGB[0..$, 0..$, 1].rcslice.ptr[0..srcH*srcW]);
    d_inputB.upload(imRGB[0..$, 0..$, 2].rcslice.ptr[0..srcH*srcW]);
    

    auto gaussianKernel = gaussian!float(2, 5, 5);

    auto d_filter_u = new CLBuffer(context, BufferMeta(FLOAT, cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1]));
    d_filter_u.upload(gaussianKernel.ptr[0..gaussianKernel.elementCount]);

    auto convR = new Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1], context);
    auto convG = new Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1], context);
    auto convB = new Convolution(cast(int)srcH, cast(int)srcW, 1,
        cast(int)gaussianKernel.shape[0], cast(int)gaussianKernel.shape[1], context);
    
    auto d_filteredR = convR.run(d_inputR, d_filter_u);
    auto d_filteredG = convG.run(d_inputG, d_filter_u);
    auto d_filteredB = convB.run(d_inputB, d_filter_u);

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
    
    imshow(imRGBI, "original");
    imshow(imGrayFloat, "imGrayFloat");
    imshow(imFiltered, "imFiltered");
    imshow(merged, "imFiltered 3");

    waitKey();
    
    destroyFigures();
}