import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))

backend = "CUDA"
nuke.pluginAppendPath(
    os.path.join(current_directory, 'plugin_libs/%s/%i_%i' % (backend, nuke.NUKE_VERSION_MAJOR,nuke.NUKE_VERSION_MINOR)))

try:
    nuke.load('pc_nuke_diffusion')
except RuntimeError:
    error_message = 'Error! Failed to load pc_nuke_diffusion library. Please check pc_nuke_diffusion installation.'
    print(error_message)
    #nuke.tprint(error_message)
