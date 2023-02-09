module preprocessor;

import mir.ndslice;
import dcv.core;
import dcv.imgproc;
import oclcv;

immutable int Hmin1 = 40, Smin1 = 0, Vmin1 = 0;
immutable int Hmax1 = 85, Smax1 = 255, Vmax1 = 255;

enum a1 = 3; // kernel size and interations of Morphology Transform
enum i1 = 1;
enum a2 = 3; // a1,i1 for OPEN, a2,i2 for CLOSE
enum i2 = 1;


class Preprocessor {

private:
    CLContext clctx;
    CLBuffer d_rgb;
    HSVConv rgb2hsvconv;
    INRANGE3 thresholder;
    MorphED eroder;
    MorphED dilater;

    Slice!(ubyte*, 3, Contiguous) hsv;
public:
    Slice!(ubyte*, 2, Contiguous) thresh1;
    
    this(int height, int width){
        hsv = slice!ubyte([height, width, 3], 0);
        thresh1 = slice!ubyte([height, width], 0);

        clctx = new CLContext;
        eroder = new MorphED(height, width, ERODE, a1, clctx);
        dilater = new MorphED(height, width, DILATE, a1, clctx);
        rgb2hsvconv = new HSVConv(height, width, RGB2HSV, clctx);
        thresholder = new INRANGE3(height, width, clctx);
        d_rgb = new CLBuffer(clctx, BufferMeta(UBYTE, height, width, 3));
    }
    
    ~this(){

    }
    
    void binarize(ref Slice!(ubyte*, 3, Contiguous) frame){
        d_rgb.upload(frame.ptr[0..frame.elementCount]);
        auto d_hsv = rgb2hsvconv.run(d_rgb);
        auto d_thresh1 = thresholder.run(d_hsv, Hmin1, Hmax1, Smin1, Smax1, Vmin1, Vmax1);
        
        //foreach (i; 0..i1)// open
        //{
            d_thresh1 = eroder.run(d_thresh1);
            d_thresh1 = dilater.run(d_thresh1);
            
        //}
        
        //foreach (i; 0..i2)// close
        //{
            d_thresh1 = dilater.run(d_thresh1);
            d_thresh1 = eroder.run(d_thresh1);
        //}
        
        d_thresh1.download(thresh1.ptr[0..thresh1.elementCount]);
    }
    
}