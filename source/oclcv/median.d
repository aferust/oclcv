module oclcv.median;

import oclcv.clcore;

import dplug.core.nogc;
import bc.string;

final class Median2D {
public:
@nogc nothrow:
    this(int height, int width, int kernelSize, CLContext ctx){

        width_ = width; height_= height; kernelSize_ = kernelSize;

        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    private bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;

        auto compilerParam = RCStringZ.from(nogcFormat!"-D T=\"uchar\" -D T1=\"uchar\" -D cn=1 -D WINDOW_SIZE=%d -D filter_offset=%d"
        (kernelSize_, kernelSize_ / 2));
        // auto compilerParam = RCStringZ.from(nogcFormat!"-D WINDOW_SIZE=%d -D filter_offset=%d"(
        //    kernelSize_, kernelSize_ / 2));
        prog_ = mallocNew!CLProgram(CTKernel.KMEDIAN, context_, compilerParam[]);

        
        if(kernelSize_ == 3){
            _kernel = prog_.getKernel("medianFilter3");
        }else if (kernelSize_ == 5){
            _kernel = prog_.getKernel("medianFilter5");
        }else if (kernelSize_ == 7){ 
            _kernel = prog_.getKernel("medianFilter7");
        } else if (kernelSize_ > 7){ // slow. todo: find better kernels
            _kernel = prog_.getKernel("MedianFilter2D");
        }
        
        return true;
    }
    
    CLBuffer run(CLBuffer d_src_mono){

        debug _assert(d_src_mono.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src_mono.metaData.numberOfChannels == 1, "Input's channel count must be 1");

        CLBuffer d_output = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 1));

        if(kernelSize_ == 3 || kernelSize_ == 5 || kernelSize_ == 7){
            int src_step = cast(int)(width_  * ubyte.sizeof);
            int dst_step = cast(int)(width_  * ubyte.sizeof);
            int src_offset = 0;
            int dst_offset = 0;
            int dst_rows = height_;
            int dst_cols = width_;
            _kernel.setArgs(d_src_mono, src_step, src_offset, d_output, dst_step, dst_offset, dst_rows, dst_cols);
        }else {
            _kernel.setArgs(d_src_mono, d_output, width_, height_);
        }

        _kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16), BlockDim(16,16));
        context_.finish(0);
        
        return d_output;
    }

private:
    int width_, height_;
    int kernelSize_;
    CLContext context_;
    CLProgram prog_;
    int TILE_SIZE;
    CLKernel _kernel;
}

final class Median3D {
public:
@nogc nothrow:
    this(int height, int width, int kernelSize, CLContext ctx){

        width_ = width; height_= height; kernelSize_ = kernelSize;

        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    private bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;

        auto compilerParam = RCStringZ.from(nogcFormat!"-D T=\"uchar3\" -D T1=\"uchar\" -D cn=3 -D KERNEL_SIZE=%d -D WINDOW_SIZE=%d -D filter_offset=%d"(
            kernelSize_, kernelSize_, kernelSize_ / 2
        ));
        prog_ = mallocNew!CLProgram(CTKernel.KMEDIAN, context_, compilerParam[]);

        if(kernelSize_ == 3){
            _kernel = prog_.getKernel("medianFilter3");
        }else if (kernelSize_ == 5){
            _kernel = prog_.getKernel("medianFilter5");
        }else if (kernelSize_ == 7){ // has kind of jagged edges, will fix
            _kernel = prog_.getKernel("medianFilter7");
        } else if (kernelSize_ > 7){ // slow. todo: find better kernels
            _kernel = prog_.getKernel("MedianFilter3D");
        }
        
        return true;
    }
    
    CLBuffer run(CLBuffer d_src3d){

        debug _assert(d_src3d.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src3d.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        CLBuffer d_output = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 3));

        if(kernelSize_ == 3 || kernelSize_ == 5 || kernelSize_ == 7){
            int src_step = cast(int)(width_ * 3 * ubyte.sizeof);
            int dst_step = cast(int)(width_ * 3 * ubyte.sizeof);
            int src_offset = 0;
            int dst_offset = 0;
            int dst_rows = height_;
            int dst_cols = width_;
            _kernel.setArgs(d_src3d, src_step, src_offset, d_output, dst_step, dst_offset, dst_rows, dst_cols);
        }else {
            _kernel.setArgs(d_src3d, d_output, width_, height_);
        }

        int tileSize = 16; 
        _kernel.launch(0, GridDim((width_ + tileSize - 1)/tileSize, (height_ + tileSize - 1)/tileSize), BlockDim(tileSize, tileSize));
        context_.finish(0);
        
        return d_output;
    }

private:
    int width_, height_;
    int kernelSize_;
    CLContext context_;
    CLProgram prog_;
    int TILE_SIZE;
    CLKernel _kernel;
}
/+
final class Median3D {
public:
@nogc nothrow:
    this(int height, int width, int kernelSize, CLContext ctx){

        width_ = width; height_= height; kernelSize_ = kernelSize;

        initialize(ctx);
    }

    ~this(){
        destroyFree(prog_);
    }

    private bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;

        auto compilerParam = RCStringZ.from(nogcFormat!"-D KERNEL_SIZE=%d -D filter_offset=%d"(
            kernelSize_, kernelSize_ / 2));
        prog_ = mallocNew!CLProgram(CTKernel.KMEDIAN, context_, compilerParam[]);
        
        _kernel = prog_.getKernel("MedianFilter3D");
        
        return true;
    }
    
    CLBuffer run(CLBuffer d_src3d){

        debug _assert(d_src3d.metaData.dataType == UBYTE, "Input type must be ubyte"); 
        debug _assert(d_src3d.metaData.numberOfChannels == 3, "Input's channel count must be 3");

        CLBuffer d_output = mallocNew!CLBuffer(context_, BufferMeta(UBYTE, height_, width_, 3));

        _kernel.setArgs(d_src3d, d_output, width_, height_);

        int tileSize = 16; 
        _kernel.launch(0, GridDim((width_ + tileSize - 1)/tileSize, (height_ + tileSize - 1)/tileSize), BlockDim(tileSize,tileSize));
        context_.finish(0);
        
        return d_output;
    }

private:
    int width_, height_;
    int kernelSize_;
    CLContext context_;
    CLProgram prog_;
    int TILE_SIZE;
    CLKernel _kernel;
}
+/