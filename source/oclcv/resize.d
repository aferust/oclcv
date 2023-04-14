module oclcv.resize;

import oclcv.clcore;

final class Resize3(string method) {
    // available methods : resize_linear, resize_bicubic
public:
    this(int srcHeight, int srcWidth, int dstWidth, int dstHeight, CLContext ctx){
        srcHeight_ = srcHeight; srcWidth_= srcWidth;
        dstHeight_ = dstHeight; dstWidth_= dstWidth;
        
        if(!initialize(ctx)){
            throw new Exception("Problem initializing the kernel");
        }
    }

    ~this(){
        destroy(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = new CLProgram(CTKernel.KRESIZE, context_);
        resize_kernel = prog_.getKernel(method);

        d_out = new CLBuffer(context_, BufferMeta(UBYTE, dstHeight_, dstWidth_, 3));
        
        return true;
    }

    CLBuffer run(CLBuffer d_src_3){
        debug _assert(d_src_3.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_3.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        struct _int2 {int x, y;}
        auto srcSize = _int2(srcWidth_, srcHeight_);
        auto dstSize = _int2(dstWidth_, dstHeight_);
        resize_kernel.setArgs(d_src_3, d_out, srcSize, dstSize);
        
        _resize();

        return d_out;
    }

    void _resize(){
        import std.algorithm.comparison : max;
        resize_kernel.launch(0, GridDim((dstWidth_ + 16 - 1)/16, (dstHeight_ + 16 - 1)/16), BlockDim(16,16));
        //size_t[3] sz = [max(srcWidth_, dstWidth_), max(srcHeight_, dstHeight_), 1];
        //resize_kernel.launch(0, null, sz.ptr, null);
        context_.finish(0);
    }

private:
    int srcWidth_, srcHeight_;
    int dstWidth_, dstHeight_;
    
    CLContext context_;
    CLProgram prog_;

    CLKernel resize_kernel;
    
    CLBuffer d_out;
}