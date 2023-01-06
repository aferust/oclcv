
import core.memory : GC;

import std.stdio;
import std.process;
import std.datetime.stopwatch : StopWatch;

import dcv.core;
import dcv.plot.figure;

import mir.ndslice;
import bindbc.opencl;

import oclcv;

// compile release for speed: dub -b release
// Video resolution
enum W = 640;
enum H = 480;

void main()
{
    auto _ret = loadOpenCL();
    writeln("Load CL: ", _ret);

    CLContext context = new CLContext;
    context.clInfo().writeln;
    
    
    // for video file as input
    /*auto pipes = pipeProcess(["ffmpeg", "-i", "file_example_MP4_640_3MG.mp4", "-f", "image2pipe",
     "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"], // yuv420p
        Redirect.stdout);*/
    // for camera device as input
    auto pipes = pipeProcess(["ffmpeg", "-y", "-hwaccel", "auto", "-f", "dshow", "-video_size", "640x480", "-i",
        `video=Lenovo EasyCamera`, "-framerate", "30", "-f", "image2pipe", "-vcodec", "rawvideo", 
        "-pix_fmt", "rgb24", "-"], Redirect.stdout);
    
    // scope(exit) wait(pipes.pid);
    auto conv = new RGB2GRAY(640, 480, context);
    auto d_rgb = new CLBuffer(context, BufferMeta(UBYTE, H, W, 3));

    // Process video frames
    auto frame = slice!ubyte([H, W, 3], 0);
    auto gray = slice!ubyte([H, W], 0);

    double fps = 30.0;
    double waitFrame = 1.0;
    StopWatch s;
    s.start;

    while(1)
    {
        import std.algorithm.comparison : max;

        s.reset;
        // Read a frame from the input pipe into the buffer
        ubyte[] dt = pipes.stdout.rawRead(frame.ptr[0..H*W*3]);
        // If we didn't get a frame of video, we're probably at the end
        if (dt.length != H*W*3) break;

        d_rgb.upload(frame.ptr);
        auto d_gray = conv.run(d_rgb);
        d_gray.download(gray.ptr);

        auto figGray = imshow(gray, "gray");
        int wait = max(1, cast(int)waitFrame - cast(int)s.peek.total!"msecs");
        
        if (waitKey(wait) == KEY_ESCAPE)
            break;

        if (!figGray.visible)
            break;
    }
    
    
}