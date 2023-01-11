module oclcv.color;

import oclcv.clcore;

final class RGB2GRAY {
public:
    this(int height, int width, CLContext ctx){
        width_ = width; height_= height;
        initialize(ctx);
    }

    ~this(){
        destroy(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = new CLProgram(CTKernel.KGRAY, context_);
        rgb2gray_kernel = prog_.getKernel("rgb2gray");

        d_gray = new CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 1));
        
        return true;
    }

    CLBuffer run(CLBuffer d_src_rgb){
        debug _assert(d_src_rgb.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_rgb.metaData.numberOfChannels == 3, "Input's channel count must be 3");

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
    
    CLBuffer d_gray;
}

// -D depth=0 -D PIX_PER_WI_Y=4
final class YUV2RGB {
public:
    this(int height, int width, CLContext ctx){
        width_ = width; height_= height;
        initialize(ctx);
    }

    ~this(){
        destroy(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = new CLProgram(CTKernel.KYUV, context_);
        
        YUV2RGB_kernel = prog_.getKernel("yuv2rgb");

        d_rgb = new CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 3));
        
        return true;
    }

    CLBuffer run(CLBuffer d_src_yuv){
        debug _assert(d_src_yuv.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_yuv.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        struct _int2 {int x, y;}
        auto sz = _int2(width_, height_);
        YUV2RGB_kernel.setArgs(d_src_yuv, d_rgb, sz);
        
        convert();

        return d_rgb;
    }

    void convert(){
        YUV2RGB_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

private:
    int width_, height_;
    CLContext context_;
    CLProgram prog_;

    CLKernel YUV2RGB_kernel;
    
    CLBuffer d_rgb;
}
