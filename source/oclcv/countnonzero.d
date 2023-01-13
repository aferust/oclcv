module oclcv.countnonzero;

import oclcv.clcore;

final class CountNonZero {
public:
    this(int height, int width, string comparisonOP, CLContext ctx){

        import std.algorithm: canFind;

        if(!["<", ">", "<=", ">=", "==", "!="].canFind(comparisonOP)){
            debug _assert(0, `unsupported OP. Please use one of these OPs: "<", ">", "<=", ">=", "==", "!="`);
        }

        width_ = width; height_= height; _comparisonOP = comparisonOP;
        initialize(ctx);
    }

    ~this(){
        destroy(d_count);
        destroy(prog_);
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = new CLProgram(CTKernel.KCOUNTNONZERO, context_, "-D OP=" ~ _comparisonOP);
        _kernel = prog_.getKernel("countNonZero");

        d_count = new CLBuffer(context_, BufferMeta(ULONG, 1, 1, 1));
        
        return true;
    }

@nogc nothrow:
    ulong run(CLBuffer d_src_mono, ubyte cval){
        debug _assert(d_src_mono.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_mono.metaData.numberOfChannels == 1, "Input's channel count must be 1");

        ulong ret;
        d_count.upload((&ret)[0..1]);
        
        int[2] sz = [width_, height_];
        _kernel.setArgs(d_src_mono, cval, d_count, sz);
        
        compute();
        d_count.download((&ret)[0..1]);

        return ret;
    }

    void compute(){
        _kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

private:
    int width_, height_;
    string _comparisonOP;
    CLContext context_;
    CLProgram prog_;

    CLKernel _kernel;
    
    CLBuffer d_count;
}