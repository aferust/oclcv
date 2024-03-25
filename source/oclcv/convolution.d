module oclcv.convolution;

import oclcv.clcore;
import core.stdc.stdlib, core.stdc.stdio;
import dplug.core.nogc;

final class Convolution {
public:
@nogc nothrow:
    this(int inputHeight, int inputWidth, int inputDepth, int filterHeight, int filterWidth, CLContext ctx){
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputDepth = inputDepth;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        
        if(!initialize(ctx)){
            printf("Problem initializing the OpenCL kernel %s", __FILE__.ptr);
            exit(-1);
        }
    }

    ~this(){
        destroyFree(prog_);
    }

    bool initialize(CLContext ctx){
        import std.conv : to;
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = mallocNew!CLProgram(CTKernel.KCONV, context_);
        _kernel = prog_.getKernel("convolution");
        
        return true;
    }

    CLBuffer run(CLBuffer d_src, CLBuffer d_filter){
        import std.algorithm.searching : canFind;
        debug _assert(d_src.metaData.dataType == FLOAT, "Input type must be ubyte"); 
        debug _assert([1,2,3].canFind(d_src.metaData.numberOfChannels), "Input's channel count must be 1,2, or 3");

        CLBuffer d_out = mallocNew!CLBuffer(context_, BufferMeta(FLOAT, inputHeight, inputWidth, inputDepth));

        _kernel.setArgs(d_src, d_filter, d_out, inputWidth, inputHeight, inputDepth, filterWidth, filterHeight);
        
        _conv();

        return d_out;
    }

    void _conv(){
        import std.algorithm.comparison : max;
        if(inputDepth == 3){
            _kernel.launch(0, GridDim((inputWidth + 16 - 1)/16, (inputHeight + 16 - 1)/16, 3), BlockDim(16,16));
        } else
            _kernel.launch(0, GridDim((inputWidth + 16 - 1)/16, (inputHeight + 16 - 1)/16), BlockDim(16,16));
        context_.finish(0);
    }

private:
    int inputHeight;
    int inputWidth;
    int inputDepth;
    int filterHeight;
    int filterWidth;
    int filterDepth;
    
    CLContext context_;
    CLProgram prog_;

    CLKernel _kernel;
}