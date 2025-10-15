#include "DDImage/PlanarIop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"
#include <memory>


// Include Stable Diffusion headers
#include "stable-diffusion.h"
#include "stable_diffusion_wrapper.h"


using namespace DD::Image;

static const char* const CLASS = "pc_sd_upscaler";
static const char* const HELP = "Stable Diffusion For Nuke";

class pc_sd_upscaler : public PlanarIop {
    const char* model_path = "";

    double format_w, format_y;
    int upscale_factor = 4;
    bool offload_params_to_cpu = false;
    bool diffusion_conv_direct  = false;

    sd_image_t upscaled_image;
    sd_images_out init_data;
    upscaler_ctx_t* upscaler_ctx;

  Format *customFormat;

public:
    pc_sd_upscaler(Node* node) : PlanarIop(node) {
        inputs(1);
        sd_set_log_callback(nuke_sd_log_cb, this);
        format_w = format_y = 512;        
    }
    const char* input_label(int input, char* buffer) const
    {
    switch (input) {
        case 0:
        return "Input Image";
    }
    }

    bool useStripes() { return false; };
    virtual bool renderFullPlanes() const { return true; };

    bool isConnected(int inputIndex) {
        return (node_input(inputIndex)!=nullptr);
    }

    void _validate(bool for_real)
    {
        if(isConnected(0)) {
            Format inpFormat = input(0)->format();
            customFormat = new Format(int(inpFormat.w()), int(inpFormat.h()));
            format_w = inpFormat.w();
            format_y = inpFormat.h();

            delete customFormat;
            customFormat = new Format(int(format_w*upscale_factor), int(format_y*upscale_factor));
            info_.format(*customFormat);
            info_.full_size_format(*customFormat);
            info_.x(0);
            info_.y(0);
            info_.r(int(format_w*upscale_factor));
            info_.t(int(format_y*upscale_factor));
            info_.turn_on(Mask_RGBA);    
        }
    }
    
    void getRequests(const Box& bbox, const DD::Image::ChannelSet& channels, int count, RequestOutput &reqData) const
    {
        reqData.request( input(0), bbox, channels, count);
    }    

    void renderStripe( ImagePlane& outputPlane){

        if (upscaler_ctx == NULL) {
            error("Failed To load upsacaler model.");
            return;
        }

        if(isConnected(0)) {
            printf("Input Image connected\n"); 
            init_data = input2sdimages(input(0),(int)format_w,(int)format_y,true);
            upscaled_image = upscale(upscaler_ctx, init_data.rgb, upscale_factor);
            free(init_data.rgb.data);
                                            
            if (upscaled_image.data == NULL) {
                error("Upsacled failed\n");
                return;
            } 
            else
            {
                printf("Upsacled Succed\n");
                outputPlane.makeWritable();        
                Box bounds = outputPlane.bounds();                
                ChannelMask channels = outputPlane.channels();
                Channel red = channels.first();
                Channel green = channels.next(red);
                Channel blue = channels.next(green);          
                int iter = 0;
                for (Box::iterator it = bounds.begin(); it != bounds.end(); it++)
                {
                    //printf("iterator - %i\n",iter);
                    float r = static_cast<float>(upscaled_image.data[iter++]);
                    float g = static_cast<float>(upscaled_image.data[iter++]);
                    float b = static_cast<float>(upscaled_image.data[iter++]);
                    outputPlane.writableAt(it.x,((int)format_y*upscale_factor)-1-it.y,outputPlane.chanNo(red)) = r/255;
                    outputPlane.writableAt(it.x,((int)format_y*upscale_factor)-1-it.y,outputPlane.chanNo(green)) = g/255;
                    outputPlane.writableAt(it.x,((int)format_y*upscale_factor)-1-it.y,outputPlane.chanNo(blue)) = b/255;
                }
                free(upscaled_image.data);
                
            }
        }
    }

    int  knob_changed(DD::Image::Knob* k)
    {
        if (k->is("reload_model") || k->is("model_path"))
        {
            if (upscaler_ctx) {
                free_upscaler_ctx(upscaler_ctx);
                upscaler_ctx = NULL;
            }
            upscaler_ctx = new_upscaler_ctx(model_path,
                                            offload_params_to_cpu,
                                            diffusion_conv_direct,
                                            get_num_physical_cores());
            if (upscaler_ctx == NULL) {
                error("Failed To load model.");
            }
            upscale_factor = get_upscale_factor(upscaler_ctx);
            Knob* t = knob("upscale_factor");
            if (t) t->set_value(upscale_factor);              
            return 1;
        }
        return PlanarIop::knob_changed(k);
    }

    void knobs(Knob_Callback f)
    {
        Button(f, "reload_model", "Reload Model");
        SetFlags(f, Knob::STARTLINE );

        File_knob(f, &model_path, "model_path", "model_path" );
        SetFlags(f, Knob::STARTLINE );
        Int_knob(f, &upscale_factor, "upscale_factor", "upscale_factor");
        SetFlags(f, Knob::READ_ONLY);

        Bool_knob(f, &offload_params_to_cpu, "offload_params_to_cpu", "offload_params_to_cpu");
        SetFlags(f, Knob::STARTLINE );
        Bool_knob(f, &diffusion_conv_direct, "diffusion_conv_direct", "diffusion_conv_direct");
        SetFlags(f, Knob::STARTLINE );

    }    

    const char* Class() const { return desc.name; };
    const char* node_help() const { return HELP; };
    static const Iop::Description desc;  
};

static Iop* build(Node* node) { 
    return new NukeWrapper( new pc_sd_upscaler(node));
}
const Iop::Description pc_sd_upscaler::desc(CLASS, "pc_nuke_diffusion", build);
