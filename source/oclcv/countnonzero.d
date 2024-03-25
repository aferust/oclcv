module oclcv.countnonzero;

import oclcv.clcore;
import std.array;
import dplug.core.nogc;
import bc.string;

final class CountNonZero {
public:
@nogc nothrow:
    this(int height, int width, string comparisonOP, CLContext ctx){

        import std.algorithm: canFind;

        if(!["<", ">", "<=", ">=", "==", "!="].staticArray[].canFind(comparisonOP)){
            debug _assert(0, `unsupported OP. Please use one of these OPs: "<", ">", "<=", ">=", "==", "!="`);
        }

        width_ = width; height_= height; _comparisonOP = comparisonOP;
        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    private bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        
        prog_ = mallocNew!CLProgram(CTKernel.KCOUNTNONZERO, context_, RCStringZ.from("-D OP=", _comparisonOP)[]);
        _kernel = prog_.getKernel("countNonZero");
        
        return true;
    }

@nogc nothrow:
    ulong run(CLBuffer d_src_mono, ubyte cval){
        debug _assert(d_src_mono.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_mono.metaData.numberOfChannels == 1, "Input's channel count must be 1");

        CLBuffer d_count = mallocNew!CLBuffer(context_, BufferMeta(ULONG, 1, 1, 1));

        ulong ret;
        d_count.upload((&ret)[0..1]);
        
        int[2] sz = [width_, height_];
        _kernel.setArgs(d_src_mono, cval, d_count, sz);
        
        compute();
        d_count.download((&ret)[0..1]);
        destroyFree(d_count);
        return ret;
    }

    private void compute(){
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
}