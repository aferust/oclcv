module oclcv.sgm;

import core.stdc.stdio : printf;

import oclcv.clcore;

final class StereoSGMCL{
public:
    this(int width, int height, int disp_size, CLContext ctx){

        width_ = width; height_= height; disp_size_ = disp_size;
        auto r = initialize(ctx);
        debug _assert(r, "error!");
    }

    bool initialize(CLContext ctx){
        if(!ctx)
            return false;
        context_ = ctx;
        //initialize kernels
        sgm_prog_ = new CLProgram(CTKernel.KSGM, context_);
        m_census_kernel = sgm_prog_.getKernel("census_kernel");
        m_matching_cost_kernel_128 = sgm_prog_.getKernel("matching_cost_kernel_128");
        m_compute_stereo_horizontal_dir_kernel_0 = sgm_prog_.getKernel("compute_stereo_horizontal_dir_kernel_0");
        m_compute_stereo_horizontal_dir_kernel_4 = sgm_prog_.getKernel("compute_stereo_horizontal_dir_kernel_4");
        m_compute_stereo_vertical_dir_kernel_2 = sgm_prog_.getKernel("compute_stereo_vertical_dir_kernel_2");
        m_compute_stereo_vertical_dir_kernel_6 = sgm_prog_.getKernel("compute_stereo_vertical_dir_kernel_6");
        m_compute_stereo_oblique_dir_kernel_1 = sgm_prog_.getKernel("compute_stereo_oblique_dir_kernel_1");
        m_compute_stereo_oblique_dir_kernel_3 = sgm_prog_.getKernel("compute_stereo_oblique_dir_kernel_3");
        m_compute_stereo_oblique_dir_kernel_5 = sgm_prog_.getKernel("compute_stereo_oblique_dir_kernel_5");
        m_compute_stereo_oblique_dir_kernel_7 = sgm_prog_.getKernel("compute_stereo_oblique_dir_kernel_7");
        m_winner_takes_all_kernel128 = sgm_prog_.getKernel("winner_takes_all_kernel128");
        m_check_consistency_left = sgm_prog_.getKernel("check_consistency_kernel_left");
        m_median_3x3 = sgm_prog_.getKernel("median3x3");
        m_copy_u8_to_u16 = sgm_prog_.getKernel("copy_u8_to_u16");
        m_clear_buffer = sgm_prog_.getKernel("clear_buffer");

        //create buffers

        d_left = new CLBuffer(context_, BufferMeta(ULONG, height_, width_));
        d_right = new CLBuffer(context_, BufferMeta(ULONG, height_, width_));
        d_matching_cost = new CLBuffer(context_, BufferMeta(UBYTE, height_, width_, disp_size_));
        d_scost = new CLBuffer(context_, BufferMeta(USHORT, height_, width_, disp_size_));
        d_left_disparity = new CLBuffer(context_, BufferMeta(USHORT, height_, width_));
        d_right_disparity = new CLBuffer(context_, BufferMeta(USHORT, height_, width_));
        d_tmp_left_disp = new CLBuffer(context_, BufferMeta(USHORT, height_, width_));
        d_tmp_right_disp = new CLBuffer(context_, BufferMeta(USHORT, height_, width_));

        //setup kernels
        
        m_matching_cost_kernel_128.setArgs(d_left, d_right, d_matching_cost, width_, height_);
        m_compute_stereo_horizontal_dir_kernel_0.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_horizontal_dir_kernel_4.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_vertical_dir_kernel_2.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_vertical_dir_kernel_6.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_oblique_dir_kernel_1.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_oblique_dir_kernel_3.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_oblique_dir_kernel_5.setArgs(d_matching_cost, d_scost, width_, height_);
        m_compute_stereo_oblique_dir_kernel_7.setArgs(d_matching_cost, d_scost, width_, height_);
        m_winner_takes_all_kernel128.setArgs(d_left_disparity, d_right_disparity, d_scost, width_, height_);
        
        m_median_3x3.setArgs(d_left_disparity, d_tmp_left_disp, width_, height_);
        m_copy_u8_to_u16.setArgs(d_matching_cost, d_scost);

        return true;
    }

    @nogc nothrow
    CLBuffer run(CLBuffer d_src_left, CLBuffer d_src_right){
        debug _assert(d_src_left.metaData.dataType == UBYTE, "left data type should be ubyte");
        debug _assert(d_src_right.metaData.dataType == UBYTE, "right data type should be ubyte");

        debug _assert(d_src_left.metaData.numberOfChannels == 1, "Only single channel images are supported");
        debug _assert(d_src_right.metaData.numberOfChannels == 1, "Only single channel images are supported");

        this.d_src_left = d_src_left;
        this.d_src_right = d_src_right;

        m_census_kernel.setArgs(d_src_left, d_left, width_, height_);
        m_check_consistency_left.setArgs(d_tmp_left_disp, d_tmp_right_disp, d_src_left, width_, height_);
        
        census();
        mem_init();
        matching_cost();
        scan_cost();
        winner_takes_all();
        median();
        context_.finish(0);
        return d_tmp_left_disp;
    }

    ~this(){
        destroy(sgm_prog_);
        
        destroy(d_left);
        destroy(d_right);
        destroy(d_matching_cost);
        destroy(d_scost);
        destroy(d_left_disparity);
        destroy(d_right_disparity);
        destroy(d_tmp_right_disp);
    }

private:

@nogc nothrow:

    void census(){
        m_census_kernel.setArgs(d_src_left, d_left);
        m_census_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        
        context_.finish(0);
        m_census_kernel.setArgs(d_src_right, d_right);
        m_census_kernel.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        context_.finish(0);
    }

    void mem_init(){
        m_clear_buffer.setArgs(d_left_disparity);
        m_clear_buffer.launch(0, GridDim(cast(int)(width_ * height_ * (ushort.sizeof)/ 32/ 256)),
                                                                        BlockDim(256));
        m_clear_buffer.setArgs(d_right_disparity);
        m_clear_buffer.launch(0, GridDim(cast(int)(width_ * height_ * ushort.sizeof/ 32/ 256)),
                                                                        BlockDim(256));
        m_clear_buffer.setArgs(d_scost);
        m_clear_buffer.launch(0, GridDim(cast(int)(width_ * height_ * ushort.sizeof * disp_size_
                                                            / 32 / 256)), BlockDim(256));
    }

    void matching_cost(){
        m_matching_cost_kernel_128.launch(0, GridDim(height_/2), BlockDim(128,2));
    }

    void scan_cost(){
        enum PATHS_IN_BLOCK = 8;
        const int obl_num_paths = width_ + height_ ;

        m_compute_stereo_horizontal_dir_kernel_0.launch(0,
        GridDim(height_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
        m_compute_stereo_horizontal_dir_kernel_4.launch(0,
        GridDim(height_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
        m_compute_stereo_vertical_dir_kernel_2.launch(0,
        GridDim(width_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
        m_compute_stereo_vertical_dir_kernel_6.launch(0,
        GridDim(width_ / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));

        m_compute_stereo_oblique_dir_kernel_1.launch(0,
        GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
        m_compute_stereo_oblique_dir_kernel_3.launch(0,
        GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
        m_compute_stereo_oblique_dir_kernel_5.launch(0,
        GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
        m_compute_stereo_oblique_dir_kernel_7.launch(0,
        GridDim(obl_num_paths / PATHS_IN_BLOCK),BlockDim(32, PATHS_IN_BLOCK));
    }

    void winner_takes_all(){
        enum WTA_PIXEL_IN_BLOCK = 8;
        m_winner_takes_all_kernel128.launch(0,
        GridDim(width_ / WTA_PIXEL_IN_BLOCK,1 * height_),
        BlockDim(32, WTA_PIXEL_IN_BLOCK));
    }

    void median(){
        m_median_3x3.setArgs(d_left_disparity, d_tmp_left_disp);
        m_median_3x3.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
        m_median_3x3.setArgs(d_right_disparity, d_tmp_right_disp);
        m_median_3x3.launch(0, GridDim((width_ + 16 - 1)/16, (height_ + 16 - 1)/16),
                                                                    BlockDim(16,16));
    }

    void check_consistency_left(){
        m_check_consistency_left.launch(0,GridDim((width_ + 16 - 1)/16,
                                              (height_ + 16 - 1)/16),BlockDim(16,16));
    }

    int width_, height_, disp_size_;
    CLContext context_;
    CLProgram sgm_prog_;

    CLKernel m_census_kernel;
    CLKernel m_matching_cost_kernel_128;

    CLKernel m_compute_stereo_horizontal_dir_kernel_0;
    CLKernel m_compute_stereo_horizontal_dir_kernel_4;
    CLKernel m_compute_stereo_vertical_dir_kernel_2;
    CLKernel m_compute_stereo_vertical_dir_kernel_6;

    CLKernel m_compute_stereo_oblique_dir_kernel_1;
    CLKernel m_compute_stereo_oblique_dir_kernel_3;
    CLKernel m_compute_stereo_oblique_dir_kernel_5;
    CLKernel m_compute_stereo_oblique_dir_kernel_7;


    CLKernel m_winner_takes_all_kernel128;

    CLKernel m_check_consistency_left;

    CLKernel m_median_3x3;

    CLKernel m_copy_u8_to_u16;
    CLKernel m_clear_buffer;

    CLBuffer d_src_left, d_src_right, d_left, d_right, d_matching_cost,
        d_scost, d_left_disparity, d_right_disparity,
        d_tmp_left_disp, d_tmp_right_disp;

}
