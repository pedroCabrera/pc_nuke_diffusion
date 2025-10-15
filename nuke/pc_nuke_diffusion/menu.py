import nuke
import os

nodelist = ["pc_sd_inference",
            "pc_sd_load_model",
            "pc_sd_nvfx_videoeffects",
            "pc_sd_upscaler"]
for node in nodelist:
    nuke.menu('Nodes').addCommand("pc_nuke_diffusion/"+node, "nuke.createNode(\""+node+"\")")
