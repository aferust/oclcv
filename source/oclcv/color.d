module oclcv.color;

import oclcv.clcore;

import dplug.core.nogc;

alias CCONV = int;
enum : CCONV {
    HSV2RGB,
    RGB2HSV,
    YUV2RGB,
    RGB2YUV
}

final class RGB2GRAY {
public:
    @nogc nothrow:

    this(int height, int width, CLContext ctx){
        width_ = width; height_= height;
        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = mallocNew!CLProgram(CTKernel.KGRAY, context_);
        rgb2gray_kernel = prog_.getKernel("rgb2gray");
        
        return true;
    }

    CLBuffer run(CLBuffer d_src_rgb){
        debug _assert(d_src_rgb.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_rgb.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        CLBuffer d_gray = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 1));
        
        struct _int2 {int x, y;}
        auto sz = _int2(width_, height_);
        rgb2gray_kernel.setArgs(d_src_rgb, d_gray, sz);
        
        convert();

        return d_gray;
    }

    void convert(){
        rgb2gray_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

private:
    int width_, height_;
    CLContext context_;
    CLProgram prog_;

    CLKernel rgb2gray_kernel;
}

final class YUVConv {
public:
@nogc nothrow:
    this(int height, int width, CCONV conversion, CLContext ctx){
        width_ = width; height_= height; _conversion = conversion;
        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = mallocNew!CLProgram(CTKernel.KYUV, context_);
        
        if(_conversion == YUV2RGB)
            conv_kernel = prog_.getKernel("yuv2rgb");
        else if (_conversion == RGB2YUV)
            conv_kernel = prog_.getKernel("rgb2yuv");
        else {
            debug _assert(0, "unsupported conversion from/to YUV!");
        }
        
        return true;
    }

    CLBuffer run(CLBuffer d_src){
        debug _assert(d_src.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        CLBuffer d_dst = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 3));

        struct _int2 {int x, y;}
        auto sz = _int2(width_, height_);
        conv_kernel.setArgs(d_src, d_dst, sz);
        
        convert();

        return d_dst;
    }

    void convert(){
        conv_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

private:
    int width_, height_;
    CCONV _conversion;

    CLContext context_;
    CLProgram prog_;

    CLKernel conv_kernel;
}

final class HSVConv {
public:
@nogc nothrow:
    this(int height, int width, CCONV conversion, CLContext ctx){
        width_ = width; height_= height; _conversion = conversion;
        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = mallocNew!CLProgram(CTKernel.KHSV, context_);
        
        if(_conversion == RGB2HSV)
            conv_kernel = prog_.getKernel("rgb2hsv");
        else if (_conversion == HSV2RGB)
            conv_kernel = prog_.getKernel("hsv2rgb");
        else {
            debug _assert(0, "unsupported conversion from/to HSV!");
        }
        
        return true;
    }

    CLBuffer run(CLBuffer d_src){
        debug _assert(d_src.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        CLBuffer d_dst = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 3));

        struct _int2 {int x, y;}
        auto sz = _int2(width_, height_);
        conv_kernel.setArgs(d_src, d_dst, sz);
        
        convert();

        return d_dst;
    }

    void convert(){
        conv_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

private:
    int width_, height_;
    CCONV _conversion;
    CLContext context_;
    CLProgram prog_;

    CLKernel conv_kernel;
}

final class INRANGE3 {
public:
@nogc nothrow:
    this(int height, int width, CLContext ctx){
        width_ = width; height_= height;
        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = mallocNew!CLProgram(CTKernel.KINRANGE3, context_);
        inRange_kernel = prog_.getKernel("inRange3");

        return true;
    }

    CLBuffer run(CLBuffer d_src_3c, ubyte lo0, ubyte hi0, ubyte lo1, ubyte hi1, ubyte lo2, ubyte hi2,
            bool inverse = false){
        debug _assert(d_src_3c.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_3c.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        CLBuffer d_bool = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 1));

        struct _int2 {int x, y;}
        auto sz = _int2(width_, height_);
        inRange_kernel.setArgs(d_src_3c, d_bool, sz, lo0, hi0, lo1, hi1, lo2, hi2, (inverse)?1:0);
        
        _threshold();

        return d_bool;
    }

    void _threshold(){
        inRange_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

private:
    int width_, height_;
    CLContext context_;
    CLProgram prog_;

    CLKernel inRange_kernel;
}