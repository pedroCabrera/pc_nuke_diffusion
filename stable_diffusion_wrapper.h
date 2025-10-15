#include <stdio.h>
#include <time.h>
#include <random>
#include <string>
#include <vector>
#include <filesystem>

#include "DDImage/PlanarIop.h"
#include "DDImage/Row.h"
#include "DDImage/Knobs.h"
#include "DDImage/NukeWrapper.h"

// #include "preprocessing.hpp"
#include "stable-diffusion.h"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_STATIC
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_STATIC
#include "stb_image_write.h"

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#define STB_IMAGE_RESIZE_STATIC
#include "stb_image_resize.h"



using namespace DD::Image;

#define SAFE_STR(s) ((s) ? (s) : "")
#define BOOL_STR(b) ((b) ? "true" : "false")


// Names for nuke knobs
static const char* weight_type_names[] = {
  "f32", "f16", "q4_0", "q4_1", "q5_0", "q5_1", "q8_0", nullptr
};
inline std::vector<const char*> get_schedule_names() {
    std::vector<const char*> names;
    for (int i = 0; i < SCHEDULE_COUNT; ++i) {
        names.push_back(sd_schedule_name(scheduler_t(i)));
    }
    names.push_back(nullptr);
    return names;
}
inline std::vector<const char*> get_sample_method_names() {
    std::vector<const char*> names;
    for (int i = 0; i < SAMPLE_METHOD_COUNT; ++i) {
        names.push_back(sd_sample_method_name(sample_method_t(i)));
    }
    names.push_back(nullptr);
    return names;
}
inline std::vector<const char*> get_rng_type_names() {
    std::vector<const char*> names;
    for (int i = 0; i < RNG_TYPE_COUNT; ++i) {
        names.push_back(sd_rng_type_name(rng_type_t(i)));
    }
    names.push_back(nullptr);
    return names;
}
inline std::vector<const char*> get_sd_type_names() {
    std::vector<const char*> names;
    for (int i = 0; i < SD_TYPE_COUNT; ++i) {
        names.push_back(sd_type_name(sd_type_t(i)));
    }
    names.push_back(nullptr);
    return names;
}

inline const char* modes_str[] = {
    "img_gen",
    "vid_gen",
    nullptr,
};

enum SDMode {
    IMG_GEN,
    VID_GEN,
    CONVERT,
    MODE_COUNT
};

inline const char* video_modes_str[] = {
    "img2video",
    "first_last",
    "video2video",
    nullptr,
};

enum SD_Video_Mode {
    IMG2VID,
    FL2VID,
    V2V,
};

struct sd_images_out {
    sd_image_t rgb;           
    sd_image_t alpha;         
    bool has_alpha = false;
};

class model_loaderNode {
public:
    virtual ~model_loaderNode() = default;
    virtual sd_ctx_t* get_ctxt() const = 0;
};

static inline uint8_t f2u8(float v) {
    if (v <= 0.f) return 0;
    if (v >= 1.f) return 255;
    return static_cast<uint8_t>(v * 255.f + 0.5f);
}

inline sd_images_out input2sdimages(Iop* input, int w, int h, bool expect_alpha = true)
{
    sd_images_out out;

    // Get Input Channels
    ChannelSet inChans = input->channels();
    
    // Prepare RGB buffer
    out.rgb.width   = w;
    out.rgb.height  = h;
    out.rgb.channel = 3;
    out.rgb.data    = (uint8_t*)malloc(out.rgb.width * out.rgb.height * out.rgb.channel);

    if (expect_alpha){
        out.alpha.width   = out.rgb.width;
        out.alpha.height  = out.rgb.height;
        out.alpha.channel = 1;
        out.alpha.data    = (uint8_t*)malloc(out.alpha.width * out.alpha.height);
        out.has_alpha     = inChans.contains(Chan_Alpha);
    }

    int iterRGB = 0;
    int iterA   = 0;
    for (int y = 0; y < out.rgb.height; ++y) {
        const int ny = out.rgb.height - 1 - y; // Flip Y
        for (int x = 0; x < out.rgb.width; ++x) {
            const float r = inChans.contains(Chan_Red)   ? input->at(x, ny, Chan_Red)   : 0.f;
            const float g = inChans.contains(Chan_Green) ? input->at(x, ny, Chan_Green) : 0.f;
            const float b = inChans.contains(Chan_Blue)  ? input->at(x, ny, Chan_Blue)  : 0.f;

            out.rgb.data[iterRGB++] = f2u8(r);
            out.rgb.data[iterRGB++] = f2u8(g);
            out.rgb.data[iterRGB++] = f2u8(b);
            if(expect_alpha){
                if (out.has_alpha)   out.alpha.data[iterA++] = f2u8(input->at(x, ny, Chan_Alpha));
                else                 out.alpha.data[iterA++] = f2u8(1.0);
            }
        }
    }
    return out;
}

inline void nuke_pretty_progress(int step, int steps, float time) {
    if (step == 0) {
        return;
    }
    std::string progress = "  |";
    int max_progress     = 50;
    int32_t current      = (int32_t)(step * 1.f * max_progress / steps);
    for (int i = 0; i < 50; i++) {
        if (i > current) {
            progress += " ";
        } else if (i == current && i != max_progress - 1) {
            progress += ">";
        } else {
            progress += "=";
        }
    }
    progress += "|";
    printf(time > 1.0f ? "\r%s %i/%i - %.2fs/it" : "\r%s %i/%i - %.2fit/s\033[K",
           progress.c_str(), step, steps,
           time > 1.0f || time == 0 ? time : (1.0f / time));
    fflush(stdout);  // for linux
    if (step == steps) {
        printf("\n");
    }
}

inline void create_output_directory(const std::string& path) {
    std::filesystem::path dir(path);
    if (!std::filesystem::exists(dir)) {
        std::error_code ec;
        std::filesystem::create_directories(dir,ec);
        if (ec) {
            printf("Failed to create directory: %s\n", path.c_str());
        }
    }
}

inline void save_image(const sd_image_t& img, const std::string output_path, bool is_jpg) {
    if (img.data == NULL) {
        return;
    }

    create_output_directory(output_path);

    std::string base_path = "nuke_sd_output";
    std::string file_ext = is_jpg ? ".jpg" : ".png";
    int last_index = -1;

    // Find the highest index in the output directory
    for (const auto& entry : std::filesystem::directory_iterator(output_path)) {
        if (entry.is_regular_file()) {
            std::string filename = entry.path().filename().string();
            // Match pattern: nuke_sd_output_<number>.ext
            size_t prefix_len = base_path.length() + 1; // +1 for '_'
            if (filename.substr(0, prefix_len) == base_path + "_") {
                size_t ext_pos = filename.rfind(file_ext);
                if (ext_pos != std::string::npos) {
                    std::string num_str = filename.substr(prefix_len, ext_pos - prefix_len);
                    try {
                        int idx = std::stoi(num_str);
                        if (idx > last_index) last_index = idx;
                    } catch (...) {}
                }
            }
        }
    }
    int next_index = last_index + 1;

    std::string final_image_path = output_path + "/" + base_path + "_" + std::to_string(next_index) + file_ext;
    if (is_jpg) {
        stbi_write_jpg(final_image_path.c_str(), img.width, img.height, img.channel,
                       img.data, 90);
        printf("save result JPEG image to '%s'\n", final_image_path.c_str());
    } else {
        stbi_write_png(final_image_path.c_str(), img.width, img.height, img.channel,
                       img.data, 0);
        printf("save result PNG image to '%s'\n", final_image_path.c_str());
    }
}

inline void nuke_sd_log_cb(enum sd_log_level_t level, const char* log, void* data) {
    int tag_color;
    const char* level_str;
    FILE* out_stream = (level == SD_LOG_ERROR) ? stderr : stdout;

    if (!log) {
        return;
    }
    switch (level) {
        case SD_LOG_DEBUG:
            level_str = "DEBUG";
            break;
        case SD_LOG_INFO:
            level_str = "INFO";
            break;
        case SD_LOG_WARN:
            level_str = "WARN";
            break;
        case SD_LOG_ERROR:
            level_str = "ERROR";
            break;
        default: /* Potential future-proofing */
            level_str = "?????";
            break;
    }

    fprintf(out_stream, "[%-5s] ", level_str);

    fputs(log, out_stream);
    fflush(out_stream);
}