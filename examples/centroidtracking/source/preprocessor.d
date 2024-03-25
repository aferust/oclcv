module preprocessor;

import mir.ndslice, mir.rc;
import oclcv;
import dplug.core;

// Hue is 0..180
immutable int Hmin1 = 52, Smin1 = 50, Vmin1 = 0;
immutable int Hmax1 = 104, Smax1 = 255, Vmax1 = 125;

/////////////////////////// set above for a desired color ////////////////////
enum a1 = 3; // kernel size and interations of Morphology Transform
enum i1 = 1;
enum a2 = 3; 
enum i2 = 1;


struct Preprocessor {
    @disable this();
private:
    CLContext clctx;
    CLBuffer d_rgb;
    HSVConv rgb2hsvconv;
    INRANGE3 thresholder;
    MorphED eroder;
    MorphED dilater;

    Slice!(RCI!ubyte, 3, Contiguous) hsv;
public:
    Slice!(RCI!ubyte, 2, Contiguous) thresh1;
    
    this(int height, int width){
        hsv = rcslice!ubyte([height, width, 3], 0);
        thresh1 = rcslice!ubyte([height, width], 0);

        clctx = mallocNew!CLContext;
        eroder = mallocNew!MorphED(height, width, ERODE, a1, clctx);
        dilater = mallocNew!MorphED(height, width, DILATE, a1, clctx);
        rgb2hsvconv = mallocNew!HSVConv(height, width, RGB2HSV, clctx);
        thresholder = mallocNew!INRANGE3(height, width, clctx);
        d_rgb = mallocNew!CLBuffer(clctx, BufferMeta(UBYTE, height, width, 3));
    }
    
    ~this(){
        destroyFree(d_rgb);
        destroyFree(thresholder);
        destroyFree(rgb2hsvconv);
        destroyFree(dilater); 
        destroyFree(eroder);
        destroyFree(clctx);
    }
    
    void binarize(ref Slice!(ubyte*, 3, Contiguous) frame){
        d_rgb.upload(frame.ptr[0..frame.elementCount]);

        auto d_hsv = rgb2hsvconv.run(d_rgb);
        scope(exit) destroyFree(d_hsv);

        auto d_thresh1 = thresholder.run(d_hsv, Hmin1, Hmax1, Smin1, Smax1, Vmin1, Vmax1);
        scope(exit) destroyFree(d_thresh1);

        auto d_eroded = eroder.run(d_thresh1);
        scope(exit) destroyFree(d_eroded);

        auto d_dilated = dilater.run(d_eroded);
        scope(exit) destroyFree(d_dilated);
        
        auto d_dilated2 = dilater.run(d_dilated);
        scope(exit) destroyFree(d_dilated2);

        auto d_eroded2 = eroder.run(d_dilated2);
        scope(exit) destroyFree(d_eroded2);
        
        d_eroded2.download(thresh1.ptr[0..thresh1.elementCount]);
    }
    
}