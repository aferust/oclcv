module oclcv.clcore;

import std.string : toStringz;
import std.conv : to;
import std.outbuffer : OutBuffer;
debug import std.stdio;

import core.stdc.stdio : printf, fread, fopen, fclose, FILE;
import core.stdc.stdlib : EXIT_FAILURE, exit, malloc, free;

import bindbc.opencl;

struct BlockDim {
    int x = 1, y = 1, z = 1;
}

struct GridDim {
    int x = 1, y = 1, z = 1;
}


alias MemFlag = int;
enum : MemFlag
{
    MEM_FLAG_READ_WRITE = 1 << 0,
    MEM_FLAG_WRITE_ONLY = 1 << 1,
    MEM_FLAG_READ_ONLY = 1 << 2,
    MEM_FLAG_USE_HOST_PTR = 1 << 3, // maybe needs to be removed, in cuda not trivial
    MEM_FLAG_ALLOC_HOST_PTR = 1 << 4,
    MEM_FLAG_COPY_HOST_PTR = 1 << 5
}

alias SyncMode = int;
enum : SyncMode
{
    SYNC_MODE_ASYNC = 0,
    SYNC_MODE_BLOCKING = 1
}

final class CLContext {
public:
    this(int platform_id = 0, int device_id = 0, int num_streams = 1){
        
        loadDLib();

        cl_platform_id p_id;
        cl_int err = 0;
        cl_uint num_platforms, num_divices;
        cl_platform_id[] p_ids;
        cl_device_id[] d_ids;

        clGetPlatformIDs(0, null, &num_platforms);
        if(num_platforms > 0){
            p_ids.length = num_platforms;
            clGetPlatformIDs(num_platforms, p_ids.ptr, null);
            if(platform_id < 0 || platform_id >= int(num_platforms)) {
                printf("Incorrect platform id %d!\n", platform_id);
                exit(EXIT_FAILURE);
            }
            p_id = p_ids[platform_id];
        }else {
            printf("Not found any platforms\n");
            exit(EXIT_FAILURE);
        }

        clGetDeviceIDs(p_id, CL_DEVICE_TYPE_ALL, 0, null, &num_divices);
        if(num_divices > 0){
            d_ids.length = num_divices;
            clGetDeviceIDs(p_id, CL_DEVICE_TYPE_ALL, num_divices,d_ids.ptr,null);
            if(device_id < 0 || device_id >= int(num_divices)){
                printf("Incorrect device id %d!\n",device_id);
                exit(EXIT_FAILURE);
            }
            cl_device_id_ = d_ids[device_id];
        }
        else{
            printf("Not found any devices\n");
            exit(EXIT_FAILURE);
        }

        cl_context_properties[3] prop = [CL_CONTEXT_PLATFORM, cast(cl_context_properties)p_id, 0];
        cl_context_ = clCreateContext(prop.ptr, 1, &cl_device_id_, null, null, &err);
        handleError(err, "creating context");
        printf("OpenCL context created! \n");

        cl_command_queues_.length = num_streams;
        for(int i=0; i < num_streams; i++){
            cl_command_queues_[i] = clCreateCommandQueue(cl_context_, cl_device_id_,
                                                        CL_QUEUE_PROFILING_ENABLE, &err);
            handleError(err, "creating ClCommandQueue");
        }

        OutBuffer oss = new OutBuffer();
        oss.writefln("Selected platform vendor: %s %s", getPlatformInfo(p_id,CL_PLATFORM_VENDOR),
                                        getPlatformInfo(p_id,CL_PLATFORM_VERSION));
        oss.writefln("Selected device name: %s", getDevInfo(cl_device_id_, CL_DEVICE_NAME));
        oss.writefln("Selected device OpenCL device version: %s",
                                    getDevInfo(cl_device_id_, CL_DEVICE_VERSION));
        oss.writefln("Selected device OpenCL C device version: %s",
                                    getDevInfo(cl_device_id_, CL_DEVICE_OPENCL_C_VERSION));
        cl_info_ = oss.toString();
    }

    ~this(){
        foreach(ref cq; cl_command_queues_)
            clReleaseCommandQueue(cq);
        clReleaseContext(cl_context_);
    }

@nogc nothrow {
    cl_context getCLContext() {return cl_context_;}
    cl_device_id getDevId() {return cl_device_id_;}
    
    
    cl_command_queue getCommandQueue(int id) {
        return cl_command_queues_[id];
    }
    
    void finish(int command_queue_id) {
        cl_int err = clFinish(cl_command_queues_[command_queue_id]);
        handleError(err, "finishing command queue");
    }

    string clInfo() {return cl_info_;}
}

private:
    string getPlatformInfo(cl_platform_id platform_id, int info_name){
        size_t info_size = 0;
        clGetPlatformInfo(platform_id, info_name, 0, null, &info_size);
        char[] str;
        str.length = info_size;
        clGetPlatformInfo(platform_id, info_name, info_size, &str[0], null);
        return str.to!string;
    }

    string getDevInfo(cl_device_id dev_id, int info_name){
        size_t info_size = 0;
        clGetDeviceInfo(dev_id, info_name, 0, null, &info_size);
        char[] str;
        str.length = info_size;
        clGetDeviceInfo(dev_id, info_name, info_size, &str[0], null);
        return str.to!string;
    }

    void loadDLib(){
        if(CLContext.support_ == CLSupport.noLibrary){
            CLContext.support_ = loadOpenCL();
            debug writeln("Load CL: ", CLContext.support_);
            if(CLContext.support_ == CLSupport.noLibrary || CLContext.support_ == CLSupport.badLibrary){
                debug _assert(0, "Problem loading opencl dynamic library");
            }
        }
        
    }

    static CLSupport support_ = CLSupport.noLibrary;
    cl_command_queue[] cl_command_queues_;
    string cl_info_;
    cl_context cl_context_;
    cl_device_id cl_device_id_;
}

alias DataType = int;
enum : DataType {
    BYTE = 0,
    UBYTE = 0,
    SHORT = 1,
    USHORT = 1,
    INT = 2,
    UINT = 2,
    FLOAT = 3,
    DOUBLE = 4,
    LONG = 5,
    ULONG = 5
}

size_t unitSize(int dtype) @nogc nothrow {
    if (dtype == -1){
        const string f = __FILE__;
        const int ln = __LINE__;
        printf("dataType of BufferMeta must be set to a supported type %s:%d", f.ptr, ln);
        exit(-1);
    } else
    if (dtype == 0){
        return byte.sizeof;
    } else
    if (dtype == 1){
        return short.sizeof;
    } else
    if (dtype == 2){
        return int.sizeof;
    } else
    if (dtype == 3){
        return float.sizeof;
    } else
    if (dtype == 4){
        return double.sizeof;
    } else
    if (dtype == 5){
        return long.sizeof;
    }

    return 0;
}
    
struct BufferMeta {
    int dataType = -1;
    size_t height;
    size_t width;
    size_t numberOfChannels = 1;

    @nogc nothrow:
    size_t memorySize(){
        return height * width * numberOfChannels * unitSize(dataType);
    }

    alias rows = height;
    alias cols = width;
}

final class CLBuffer {
public:
    this(CLContext ctx, BufferMeta buffer_meta, MemFlag flag = MEM_FLAG_READ_WRITE,
             void[] host_data = null)
    {
        meta_data = buffer_meta;

        this(ctx, flag, host_data);
    }
    
    private this(CLContext ctx, MemFlag flag = MEM_FLAG_READ_WRITE,
             void[] host_data = null){
        context_ = ctx;
        flag_ = flag;

        cl_int err;
        buffer_ = clCreateBuffer(context_.getCLContext(), getCLMemFlag(flag),
                                                        size_, null, &err);
        handleError(err, "creating buffer");
        if(host_data){
            upload(host_data, SYNC_MODE_BLOCKING, 0);
        }
    }
    
    ~this(){
        if(buffer_){
            cl_int err = clReleaseMemObject(buffer_) ;
            handleError(err, "in releasing buffer");
            buffer_ = null;
        }
    }

@nogc nothrow:
    // validate device memory for debug purpose
    int validate(){
        //check if valid mem object,
        cl_mem_object_type mem_type = 0;
        clGetMemObjectInfo(buffer_, CL_MEM_TYPE, cl_mem_object_type.sizeof, &mem_type, null);
        if (mem_type != CL_MEM_OBJECT_BUFFER)
        {
            debug writeln("CL_INVALID_MEM_OBJECT");
            return CL_INVALID_MEM_OBJECT;
        }
        //check if mem object have valid required size
        if (size_ > 0)
        {
            size_t current_size;
            clGetMemObjectInfo(buffer_, CL_MEM_SIZE,
                                size_t.sizeof, &current_size, null);
            debug writefln("[CLBuffer.validate] Buffer size: %s bytes. Required size: %s",
                current_size, size_);
            
            if (current_size < size_)
                return CL_INVALID_IMAGE_SIZE;
        }
    
        return CL_SUCCESS;
    }

    bool isNull(){return buffer_ == null;}
    
    void upload(const(void)[] data, SyncMode block_queue = SYNC_MODE_BLOCKING,
               int command_queue = 0){
        debug _assert(data.length == metaData().memorySize(), "Mismatch in source and destination memory sizes.");
        upload(data.ptr, 0, size_, block_queue, command_queue);
    }

    void download(void[] data, SyncMode block_queue = SYNC_MODE_BLOCKING,
              int command_queue = 0){
        debug _assert(data.length == metaData().memorySize(), "Mismatch in source and destination memory sizes.");
        download(data.ptr, 0, size_, block_queue, command_queue);
    }

    cl_mem getCObject(){
        return buffer_;
    }

    BufferMeta metaData(){return meta_data;}

private:

    void upload(const void* data, size_t offset, size_t size,
               SyncMode block_queue,int command_queue){
        
        cl_bool b_Block = (block_queue == SYNC_MODE_BLOCKING) ? CL_TRUE : CL_FALSE;
        cl_int err = clEnqueueWriteBuffer(context_.getCommandQueue(command_queue),
                                        buffer_, b_Block, offset, size, data, 0,
                                                                null, null);
        handleError(err, "enqueuing writing buffer");
    }

    void download(void* data, size_t offset, size_t size, SyncMode block_queue,
              int command_queue){
        cl_bool b_Block = (block_queue == SYNC_MODE_BLOCKING) ? CL_TRUE : CL_FALSE;
        cl_int err = clEnqueueReadBuffer(context_.getCommandQueue(command_queue),
                                        buffer_, b_Block, offset, size, data, 0,
                                                                null, null);
        handleError(err, "enqueuing reading buffer");
    }

    CLContext context_;
    cl_mem buffer_;
    MemFlag flag_;

    BufferMeta meta_data;
    size_t size_(){return meta_data.memorySize;}
    
}

final class CLKernel {
public:
    this(CLContext context, cl_program program, string kernel_name){
        cl_int err = CL_SUCCESS;
        context_ = context;
        kernel_name_ = kernel_name;
        kernel_ = clCreateKernel(program, kernel_name_.toStringz, &err);
        handleError(err, "creating kernel: " ~ kernel_name);
    }
    ~this(){
        cl_int err = clReleaseKernel(kernel_);
        handleError(err, "releasing kernel objects");
    }

@nogc nothrow:

    void launch(int queue_id, GridDim gd, BlockDim bd){
        size_t[3] global_w_offset = [0, 0, 0];
        size_t[3] global_w_size = [
                        size_t(gd.x * bd.x),
                        size_t(gd.y * bd.y),
                        size_t(gd.z * bd.z)];
        size_t[3] local_w_size = [size_t(bd.x),size_t(bd.y),size_t(bd.z)];

        cl_int err = clEnqueueNDRangeKernel(context_.getCommandQueue(queue_id),
                                        kernel_, 3, global_w_offset.ptr, global_w_size.ptr,
                                        local_w_size.ptr, 0, null, null);

        handleError(err, "enqueuing kernel");
    }

    void launch(int queue_id, size_t* gwo, size_t* gws, size_t* lws){
        cl_int err = clEnqueueNDRangeKernel(context_.getCommandQueue(queue_id),
                                        kernel_, 3, gwo, gws, lws, 0, null, null);
        handleError(err, "enqueuing kernel");
    }

    void setArgs(Args...)(Args args){
        import std.stdio;
        cl_int err = CL_SUCCESS;
        foreach(i, arg; args){
            
            static if(is(typeof(arg)==CLBuffer)){
                auto raw_mem = arg.getCObject();
                err = clSetKernelArg(kernel_, cl_uint(i), cl_mem.sizeof, cast(void*)&raw_mem);
            } else {
                err = clSetKernelArg(kernel_, cl_uint(i), typeof(arg).sizeof, cast(void*)&arg);
            }
            
            debug handleError(err, "setting kernel arguments of " ~ kernel_name_);
        }
    }

private:
    string kernel_name_;
    cl_kernel kernel_;
    CLContext context_;
}

final class CLProgram{
public:
    this(string source_path = "", CLContext context=null
                          , string compilation_options="-I \"./\""){
        context_ = context;

        enum MAX_SOURCE_SIZE = 0x100000;

        FILE *fp;
        char *source_str;
        size_t source_size;
    
        fp = fopen(source_path.toStringz, "r");
        if (!fp) {
            printf("Failed to load kernel.\n");
            exit(1);
        }
        source_str = cast(char*)malloc(MAX_SOURCE_SIZE);
        scope(exit) free(source_str);
        source_size = fread( source_str, 1, MAX_SOURCE_SIZE, fp);
        fclose( fp );

        createProgram(source_str, source_size, compilation_options);
        createKernels();
    }

    this(CTKernel ct_kernel, CLContext context=null
                          , string compilation_options="-I \"./\""){
        context_ = context;

        char* source_str = ct_kernel.dup.ptr;

        createProgram(source_str, ct_kernel.length, compilation_options);
        createKernels();
    }

    ~this(){
        foreach(ref item; kernels_.byValue()){
            item.destroy();
            item = null;
        }
        if(cl_program_ !is null){
            cl_int err = clReleaseProgram(cl_program_);
            handleError(err, "releasing program");
        }
    }
    bool createProgram(const char* source, size_t source_size, string compilation_options = null){
        cl_int err;
        char[] cop = null;
        if(compilation_options){
            //debug writeln(compilation_options);
            cop = compilation_options.dup;
        }
        const size_t size_src = source_size;
        
        cl_program_ = clCreateProgramWithSource(context_.getCLContext(), 1,
                                                cast(const char **)&source, &size_src, &err);
        handleError(err, "creating program with source data");
        cl_device_id dev_id = context_.getDevId();
        err = clBuildProgram(cl_program_, 1, &dev_id, cop.ptr, null, null);

        import std.format, std.conv : to;
        handleError(err, format("building program with source: %s\nUsing compilation options: %s\n", source.to!string, compilation_options));
        return true;
    }

    bool createKernels(){
        cl_uint num_kernels = 0;
        cl_int err;
        err = clCreateKernelsInProgram(cl_program_, 0, null, &num_kernels);
        if(num_kernels == 0)
            err = CL_INVALID_BINARY;
        if(err != CL_SUCCESS){
            char[] build_log;
            size_t log_size = 0;
            clGetProgramBuildInfo(cl_program_, context_.getDevId(),
                                CL_PROGRAM_BUILD_LOG, 0, null, &log_size);
            build_log.length = log_size;
            clGetProgramBuildInfo(cl_program_, context_.getDevId(),
                                CL_PROGRAM_BUILD_LOG, log_size, build_log.ptr, null);
            printf("%s \n", build_log.ptr);
            handleError(err,"creating kernels");

        }
        return true;
    }

    CLKernel getKernel(string kernel_name){
        CLKernel kernel;

        if (auto kernptr = kernel_name in kernels_)
            kernel = *kernptr;
        else{
            kernel = new CLKernel(context_, cl_program_, kernel_name);
            kernels_[kernel_name] = kernel;
        }

        if(kernel is null){
            printf("kernel has been deleted or failed to create!\n");
            exit(EXIT_FAILURE);
        }

        return kernel;
    }

    void setCLContext(CLContext context) {context_ = context;}

private:
    CLContext context_;
    cl_program cl_program_;
    CLKernel[string] kernels_;
}

@nogc nothrow:

static void handleError()(cl_int err, auto ref string msg, string f = __FILE__, int l = __LINE__){
    if(err != CL_SUCCESS){
        immutable(string) oerr = errorNumberToString(err);
        printf("[OpenCL Error] in %s !: %s %s:%d\n", msg.ptr, oerr.ptr, f.ptr, l);
        exit(EXIT_FAILURE);
    }
}

immutable(string) errorNumberToString(cl_int errorNumber)
{
    switch (errorNumber)
    {
        case CL_SUCCESS:
            return "CL_SUCCESS";
        case CL_DEVICE_NOT_FOUND:
            return "CL_DEVICE_NOT_FOUND";
        case CL_DEVICE_NOT_AVAILABLE:
            return "CL_DEVICE_NOT_AVAILABLE";
        case CL_COMPILER_NOT_AVAILABLE:
            return "CL_COMPILER_NOT_AVAILABLE";
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
            return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
        case CL_OUT_OF_RESOURCES:
            return "CL_OUT_OF_RESOURCES";
        case CL_OUT_OF_HOST_MEMORY:
            return "CL_OUT_OF_HOST_MEMORY";
        case CL_PROFILING_INFO_NOT_AVAILABLE:
            return "CL_PROFILING_INFO_NOT_AVAILABLE";
        case CL_MEM_COPY_OVERLAP:
            return "CL_MEM_COPY_OVERLAP";
        case CL_IMAGE_FORMAT_MISMATCH:
            return "CL_IMAGE_FORMAT_MISMATCH";
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
            return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
        case CL_BUILD_PROGRAM_FAILURE:
            return "CL_BUILD_PROGRAM_FAILURE";
        case CL_MAP_FAILURE:
            return "CL_MAP_FAILURE";
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
            return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
        case CL_INVALID_VALUE:
            return "CL_INVALID_VALUE";
        case CL_INVALID_DEVICE_TYPE:
            return "CL_INVALID_DEVICE_TYPE";
        case CL_INVALID_PLATFORM:
            return "CL_INVALID_PLATFORM";
        case CL_INVALID_DEVICE:
            return "CL_INVALID_DEVICE";
        case CL_INVALID_CONTEXT:
            return "CL_INVALID_CONTEXT";
        case CL_INVALID_QUEUE_PROPERTIES:
            return "CL_INVALID_QUEUE_PROPERTIES";
        case CL_INVALID_COMMAND_QUEUE:
            return "CL_INVALID_COMMAND_QUEUE";
        case CL_INVALID_HOST_PTR:
            return "CL_INVALID_HOST_PTR";
        case CL_INVALID_MEM_OBJECT:
            return "CL_INVALID_MEM_OBJECT";
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
            return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
        case CL_INVALID_IMAGE_SIZE:
            return "CL_INVALID_IMAGE_SIZE";
        case CL_INVALID_SAMPLER:
            return "CL_INVALID_SAMPLER";
        case CL_INVALID_BINARY:
            return "CL_INVALID_BINARY";
        case CL_INVALID_BUILD_OPTIONS:
            return "CL_INVALID_BUILD_OPTIONS";
        case CL_INVALID_PROGRAM:
            return "CL_INVALID_PROGRAM";
        case CL_INVALID_PROGRAM_EXECUTABLE:
            return "CL_INVALID_PROGRAM_EXECUTABLE";
        case CL_INVALID_KERNEL_NAME:
            return "CL_INVALID_KERNEL_NAME";
        case CL_INVALID_KERNEL_DEFINITION:
            return "CL_INVALID_KERNEL_DEFINITION";
        case CL_INVALID_KERNEL:
            return "CL_INVALID_KERNEL";
        case CL_INVALID_ARG_INDEX:
            return "CL_INVALID_ARG_INDEX";
        case CL_INVALID_ARG_VALUE:
            return "CL_INVALID_ARG_VALUE";
        case CL_INVALID_ARG_SIZE:
            return "CL_INVALID_ARG_SIZE";
        case CL_INVALID_KERNEL_ARGS:
            return "CL_INVALID_KERNEL_ARGS";
        case CL_INVALID_WORK_DIMENSION:
            return "CL_INVALID_WORK_DIMENSION";
        case CL_INVALID_WORK_GROUP_SIZE:
            return "CL_INVALID_WORK_GROUP_SIZE";
        case CL_INVALID_WORK_ITEM_SIZE:
            return "CL_INVALID_WORK_ITEM_SIZE";
        case CL_INVALID_GLOBAL_OFFSET:
            return "CL_INVALID_GLOBAL_OFFSET";
        case CL_INVALID_EVENT_WAIT_LIST:
            return "CL_INVALID_EVENT_WAIT_LIST";
        case CL_INVALID_EVENT:
            return "CL_INVALID_EVENT";
        case CL_INVALID_OPERATION:
            return "CL_INVALID_OPERATION";
        case CL_INVALID_GL_OBJECT:
            return "CL_INVALID_GL_OBJECT";
        case CL_INVALID_BUFFER_SIZE:
            return "CL_INVALID_BUFFER_SIZE";
        case CL_INVALID_MIP_LEVEL:
            return "CL_INVALID_MIP_LEVEL";
        default:
            return "Unknown error";
    }
}

static int getCLMemFlag(MemFlag mem_flag)
{
    int ret = 0;
    switch(mem_flag){
       case MEM_FLAG_READ_WRITE:
           ret = ret | CL_MEM_READ_WRITE; break;
       case MEM_FLAG_READ_ONLY:
           ret = ret | CL_MEM_READ_ONLY; break;
       case MEM_FLAG_WRITE_ONLY:
           ret = ret | CL_MEM_WRITE_ONLY; break;
       case MEM_FLAG_USE_HOST_PTR:
           ret = ret | CL_MEM_USE_HOST_PTR; break;
       case MEM_FLAG_ALLOC_HOST_PTR:
           ret = ret | CL_MEM_ALLOC_HOST_PTR; break;
       case MEM_FLAG_COPY_HOST_PTR:
           ret = ret | CL_MEM_COPY_HOST_PTR; break;
       default: break;
    }
    return ret;
}

void _assert(bool condition, string msg, string file = __FILE__, int line = __LINE__)
{
    if(!condition)
    {
        printf("%s %s:%d\n", msg.ptr, file.ptr, line);
        exit(EXIT_FAILURE);
    }
}

// registering library kernels
enum CTKernel {
    KGRAY = import("gray.cl"),
    KYUV = import("yuv.cl"),
    KHSV = import("hsv.cl"),
    KSGM = import("sgm.cl"),
    KINRANGE3 = import("inrange3.cl"),
    KCOUNTNONZERO = import("countnonzero.cl"),
    KMORPHED = import("morphed.cl"),
    KRESIZE = import("resize.cl")
}