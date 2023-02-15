
import std.stdio;
import std.process;
import std.container.array, std.array : staticArray;
import std.datetime.stopwatch : StopWatch;

import dcv.core;
import dcv.plot;
import dcv.measure;
import dcv.tracking.centroidtracker;

import mir.ndslice;
import oclcv;


import preprocessor;

// compile release for speed: dub -b release
// Video resolution
enum W = 640;
enum H = 480;
enum DSHOW_DEV_NAME = `video=` ~ `Lenovo EasyCamera`;

enum minArea = 2000; // min area to confirm these are objects 
enum minObject = 5000; // min area to ensure removed-object
enum smallObj = 11000; //11000; // small Object
enum colorPercent = 30;  // ratio percent of RED
enum detectBorder = false;

///////////////////////////////// set above /////////////////////////////////////////////////////

import std.conv : to;
enum WS = W.to!string;
enum HS = H.to!string;
enum SIZE_STR = WS ~ "x" ~ HS;

void main()
{
    auto centroidTracker = CentroidTracker(20);
    auto prep = Preprocessor(H, W);

    // for video file as input
    /*auto pipes = pipeProcess(["ffmpeg", "-i", "file_example_MP4_640_3MG.mp4", "-f", "image2pipe",
     "-vcodec", "rawvideo", "-pix_fmt", "rgb24", "-"],
        Redirect.stdout);*/
    // for camera device as input
    auto pipes = pipeProcess(["ffmpeg", "-y", "-hwaccel", "auto", "-f", "dshow", "-video_size", SIZE_STR, "-i",
        DSHOW_DEV_NAME, "-framerate", "30", "-f", "image2pipe", "-vcodec", "rawvideo", 
        "-pix_fmt", "rgb24", "-"], Redirect.stdout);
    
    
    // Process video frames
    auto frame = slice!ubyte([H, W, 3], 0);
    auto thr1 = slice!ubyte([H, W], 0);
    
    
    auto figDetection = imshow(frame, "Detection");
    // auto figThr1 = imshow(thr1, "thr1"); // to debug preprocessing

    double fps = 30.0;
    double waitFrame = 1.0;
    StopWatch s;
    s.start;

    while(1)
    {
        import std.algorithm.comparison : max;

        s.reset;
        // Read a frame from the input pipe into the buffer
        const ubyte[] dt = pipes.stdout.rawRead(frame.ptr[0..H*W*3]);
        // If we didn't get a frame of video, we're probably at the end
        if (dt.length != H*W*3) break;

        figDetection.draw(frame, ImageFormat.IF_RGB); // redraw the window

        immutable xLen = W;
        immutable yLen = H;

        prep.binarize(frame);
        // figThr1.draw(prep.thresh1, ImageFormat.IF_MONO);

        Array!Box boxes;
        scope(exit) boxes.clear;

        auto contours_hier = findContours(prep.thresh1);
        auto contours = contours_hier[0];
        auto hierarchy = contours_hier[1];

        import mir.math.stat: mean;
        if (contours.length){
            auto indNoHoles = indicesWithoutHoles(hierarchy);
            foreach(ci; indNoHoles){
                auto contour = contours[ci];
                
                const int area = cast(int)(contour.contourArea/1000.0)*1000;
                if (area < minArea)
                    continue;
                bool flagBorder = false;
                foreach(j; 0..contour.shape[0]){
                    const flgx = (contour[j, 0] == 0) || (contour[j, 0] == xLen);
                    const flgy = (contour[j, 1] == 0) || (contour[j, 1] == yLen);
                    if (flgx || flgy){
                        flagBorder = true;
                        break;
                    }
                }

                if (flagBorder == true && detectBorder)
                    continue;
                const Rectangle rect = boundingBox(contour);
                const xj = cast(int)rect.x;
                const yj = cast(int)rect.y;
                const wj = cast(int)rect.width;
                const hj = cast(int)rect.height;
                
                auto thresh_i = prep.thresh1[yj..yj+hj, xj..xj+wj];
                
                auto ret_i = findContours(thresh_i);
                auto contours_i = ret_i[0];

                ubyte t;
                
                if (contours_i.length){
                    ulong si = 0;
                    foreach(cnt; contours_i){
                        const sii = contourArea(cnt);
                        si += cast(ulong)(sii);
                    }
                    si = cast(ulong)(cast(float)si/1000.0f)*1000;
                    const int ratio_i = cast(int)(cast(int)(cast(float)si/area*100)/5.0f)*5;
                    t = typeObj(ratio_i, area);
                }else{
                    t = typeObj(0, area);
                }
                
                //if((t == 1) || (t == 3)) // filter your target objects
                    boxes ~= [xj, yj, xj + wj, yj + hj].staticArray!int;
            }
        }
        
        auto objects = centroidTracker.update(boxes);
        
        if (!objects.empty()) {
            foreach (obj; objects) {

                //figDetection.drawCircle(PlotCircle(cast(float)obj[1][1], cast(float)obj[1][0], 36.0f), plotRed, false, 2.0f);
                immutable box = obj.box;
                PlotPoint[2] rect = [
                    PlotPoint(cast(float)box[1], cast(float)box[0]),
                    PlotPoint(cast(float)(box[3]), cast(float)(box[2]))
                ];
                
                figDetection.drawRectangle(rect, plotRed, 2.0f);
                /*string ID = std::to_string(get<0>(obj));
                cv::putText(cameraFrame, ID, Point(get<1>(obj).first - 10, get<1>(obj).second - 10),
                            FONT_HERSHEY_COMPLEX, 0.5, Scalar(0, 255, 0), 2);*/
                
                
                /*rectangle(cameraFrame, Point(get<2>(obj)[0], get<2>(obj)[1])
                                     , Point(get<2>(obj)[2], get<2>(obj)[3]),
                                     Scalar(0, 255, 0), 2);*/
            }
            
            import core.stdc.math : sqrtf;
            
            //drawing the path
            foreach (obj; objects) {
                int k = 1;
                auto path = centroidTracker.pathKeeper[obj.id];
                for (auto i = 1; i < path.length; i++) {
                    float thickness = sqrtf(20.0f / cast(float)(k + 1) * 2.5f);
                    auto first = path[i - 1];
                    auto next  = path[i];
                    figDetection.drawLine(
                            PlotPoint(first.y, first.x), PlotPoint(next.y,
                            next.x), plotBlue, thickness);
                }
            }
        }
        
        const wait = max(1, cast(int)waitFrame - cast(int)s.peek.total!"msecs");
        

        if (waitKey(wait) == KEY_ESCAPE)
            break;

        if (!figDetection.visible/* && !figThr1.visible*/)
            break;
    }
    
    
}

ubyte typeObj(const int r, const ref int area){
    if (r >= colorPercent){
        if (area > smallObj)
            return 1;
        else
            return 2;
    }else{
        if (area > smallObj)
            return 3;
        else{
            if (area > minObject)
                return 4;
            else
                return 5;
        }
    }
}

RCArray!int indicesWithoutHoles(H)(const ref H hierarchy){
    import mir.appender;
    auto _ret = scopedBuffer!int;
    foreach (h; hierarchy)
    {
        if(h.border.seq_num - 2 >= 0 && h.border.border_type == 2 )
            _ret.put(h.border.seq_num - 2);
        
    }
    return rcarray!int(_ret.data);

}