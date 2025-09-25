#include "DDImage/PlanarIop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"
#include <memory>

// Include Stable Diffusion headers
#include "NvidiaMaxineWrapper.h"
#include "NvCVImage.h"   // for NvCVImage + pixelFormat enums

using namespace DD::Image;

static const char* const CLASS = "pc_sd_nvfx_videoeffects";
static const char* const HELP = "Stable Diffusion For Nuke";

const char* effect_names[] = {"ArtifactReduction",
                            "SuperRes",
                            "Upscale",
                            "GreenScreen",
                            "BackgroundBlur",
                            "Denoising",
                            nullptr};
const char* mode_names[] = {"conservative",
                            "aggresive",
                            nullptr};

class pc_sd_nvfx_videoeffects : public PlanarIop {
    const char* model_path = "C:/Program Files/NVIDIA Corporation/NVIDIA Video Effects/models";
    int format_w, format_y;
    int upscale_factor = 4;
    int effect = 0;
    int mode = 0;
    float strength = 1.0f;

    FXApp::Err  fxErr = FXApp::errNone;
    int         nErrs;
    FXApp       app;   

    Format *customFormat;

public:
    pc_sd_nvfx_videoeffects(Node* node) : PlanarIop(node) {
        inputs(1);
        format_w = format_y = 512;        
    }
    const char* input_label(int input, char* buffer) const
    {
    switch (input) {
        case 0:
        return "Input Image";
    }
    }
    //int minimum_inputs() const { return 1; }

    bool useStripes() { return false; };
    virtual bool renderFullPlanes() const { return true; };

    bool isConnected(int inputIndex) {
        return (node_input(inputIndex)!=nullptr);
    }

    void _validate(bool /*for_real*/) override
    {
        if (!isConnected(0)) return;

        const Format inFmt = input(0)->format();
        format_w = inFmt.w();
        format_y = inFmt.h();

        const char* fx = effect_names[effect];
        const bool scales =
            (strcmp(fx, "Upscale") == 0) || (strcmp(fx, "SuperRes") == 0);

        const int outW = scales ? format_w * upscale_factor : format_w;
        const int outH = scales ? format_y * upscale_factor : format_y;

        delete customFormat;
        customFormat = new Format(outW, outH);

        info_.format(*customFormat);
        info_.full_size_format(*customFormat);
        info_.x(0); info_.y(0);
        info_.r(outW); info_.t(outH);
        info_.turn_on(Mask_RGBA);
    }
    
    void getRequests(const Box& bbox, const DD::Image::ChannelSet& channels, int count, RequestOutput &reqData) const
    {
        reqData.request( input(0), bbox, channels, count);
    }    

    void renderStripe(ImagePlane& outputPlane) override
    {
        if (!isConnected(0)) { error("No input."); return; }
        // Recreate the effect if selection changed or not yet created
        const char* want = effect_names[effect];
        if (!app._eff || !app._effectName || strcmp(app._effectName, want) != 0) {
            app.destroyEffect();
        }
        // Set parameters
        app.FLAG_strength      = strength;
        app.FLAG_mode          = mode;
        app.FLAG_upscaleFactor = upscale_factor;
        // Create the effect if needed
        if (!app._eff) {
            fxErr = app.createEffect(want, model_path);
            if (fxErr != FXApp::errNone) {
                error("Failed to create effect '%s': %s", want, app.errorStringFromCode(fxErr));
                return;
            }
        }
        // Run the effect
        fxErr = app.processImage(input(0));
        if (fxErr != FXApp::errNone) {
            error("NvVFX processing failed: %s", app.errorStringFromCode(fxErr));
            return;
        }

        // Write out to nuke image plane
        outputPlane.makeWritable();
        if (!NvCVToImagePlane(app._dstVFX, outputPlane)) {
            error("Unsupported NvCV output format (expect RGBA8 or BGR8 interleaved).");
            return;
        }
    }

    void knobs(Knob_Callback f)
    {
        Enumeration_knob(f, &effect, effect_names, "effect", "effect");
        Enumeration_knob(f, &mode, mode_names, "mode", "mode");

        File_knob(f, &model_path, "model_path", "model_path" );
        Int_knob(f, &upscale_factor, "upscale_factor", "upscale_factor");
        SetFlags(f, Knob::SLIDER);
        Float_knob(f, &strength, "strength", "strength");
        SetFlags(f, Knob::SLIDER);
    }

    const char* Class() const { return desc.name; };
    const char* node_help() const { return HELP; };
    static const Iop::Description desc;  
};

static Iop* build(Node* node) { 
    return new NukeWrapper( new pc_sd_nvfx_videoeffects(node));
}
const Iop::Description pc_sd_nvfx_videoeffects::desc(CLASS, "pc_nuke_diffusion", build);
