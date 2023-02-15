module preprocessor;

import mir.ndslice;
import oclcv;

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
        
        //foreach (i; 0..i1) // morph opening
        //{
            d_thresh1 = eroder.run(d_thresh1);
            d_thresh1 = dilater.run(d_thresh1);
            
        //}
        
        //foreach (i; 0..i2) // morp closing
        //{
            d_thresh1 = dilater.run(d_thresh1);
            d_thresh1 = eroder.run(d_thresh1);
        //}
        
        d_thresh1.download(thresh1.ptr[0..thresh1.elementCount]);
    }
    
}