module oclcv.convolution;

import oclcv.clcore;

final class Convolution {
public:
    this(int inputHeight, int inputWidth, int inputDepth, int filterHeight, int filterWidth, CLContext ctx){
        this.inputHeight = inputHeight;
        this.inputWidth = inputWidth;
        this.inputDepth = inputDepth;
        this.filterHeight = filterHeight;
        this.filterWidth = filterWidth;
        
        if(!initialize(ctx)){
            throw new Exception("Problem initializing the kernel");
        }
    }

    ~this(){
        destroy(prog_);
    }

    bool initialize(CLContext ctx){
        import std.conv : to;
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = new CLProgram(CTKernel.KCONV, context_);
        _kernel = prog_.getKernel("convolution");
        
        d_out = new CLBuffer(context_, BufferMeta(FLOAT, inputHeight, inputWidth, inputDepth));
        
        return true;
    }

    CLBuffer run(CLBuffer d_src, CLBuffer d_filter){
        import std.algorithm.searching : canFind;
        debug _assert(d_src.metaData.dataType == FLOAT, "Input type must be ubyte"); 
        debug _assert([1,2,3].canFind(d_src.metaData.numberOfChannels), "Input's channel count must be 1,2, or 3");

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
    
    CLBuffer d_out;
}