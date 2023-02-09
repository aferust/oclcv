module oclcv.morphed;

import oclcv.clcore;

alias MORPH_OP = int;
enum : MORPH_OP {
    ERODE,
    DILATE
}

final class MorphED {
public:
    this(int height, int width, MORPH_OP op, int kernelSize, CLContext ctx){

        width_ = width; height_= height; _op = op; kernelSize_ = kernelSize;

        initialize(ctx);
    }

    ~this(){
        destroy(prog_);
    }

    private bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;

        d_output = new CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 1));
        d_tmp = new CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 1));


        tile_w = width_;
        tile_h = 1;

        block2 = BlockDim(tile_w + (2 * kernelSize_), tile_h);

        import std.conv : to;
        prog_ = new CLProgram(CTKernel.KMORPHED, context_,
                    "-D " ~ ((_op==ERODE)?"ERODE":"DILATE") ~ 
                    " -D SHARED_SIZE=" ~ to!string(block2.y*block2.x) ~
                    " -D radio="~ to!string(kernelSize_));
        //_kernel = prog_.getKernel("morphed");
        _kernel_step1 = prog_.getKernel("morphSharedStep1");
        _kernel_step2 = prog_.getKernel("morphSharedStep2");
        
        return true;
    }

@nogc nothrow:
    CLBuffer run(CLBuffer d_src_mono){

        import core.stdc.math : ceil;

        debug _assert(d_src_mono.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_mono.metaData.numberOfChannels == 1, "Input's channel count must be 1");

        tile_w = width_;
        tile_h = 1;

        block2 = BlockDim(tile_w + (2 * kernelSize_), tile_h);
        
        _kernel_step1.setArgs(d_src_mono, d_tmp, width_, height_, tile_w, tile_h);
        
        _kernel_step1.launch(0, GridDim(cast(int)ceil(cast(float)width_ / tile_w), 
                                  cast(int)ceil(cast(float)height_ / tile_h)),
                                  block2);
        context_.finish(0);

        tile_w = 8;
        tile_h = 64;

        _kernel_step2.setArgs(d_tmp, d_output, width_, height_, tile_w, tile_h);
        
        _kernel_step2.launch(0, 
            GridDim(cast(int)ceil(cast(float)width_ / tile_w), cast(int)ceil(cast(float)height_ / tile_h)),
                        BlockDim(tile_w, tile_h + (2 * kernelSize_)));
        context_.finish(0);

        return d_output;
    }

private:
    int width_, height_;
    int kernelSize_;
    int tile_w;
    int tile_h;
    BlockDim block2;
    CLContext context_;
    CLProgram prog_;
    CLBuffer d_tmp;
    CLBuffer d_output;
    //CLKernel _kernel;
    CLKernel _kernel_step1;
    CLKernel _kernel_step2;
    MORPH_OP _op;
}