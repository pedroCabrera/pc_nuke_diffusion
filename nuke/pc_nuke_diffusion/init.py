import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
nuke.pluginAppendPath(
    os.path.join(current_directory, 'plugin_libs'))
nuke.pluginAppendPath(
    os.path.join(current_directory, 'plugin_libs/Release'))
try:
    nuke.load('pc_nuke_diffusion')
except RuntimeError:
    error_message = 'Error! Failed to load pc_nuke_diffusion library. Please check pc_nuke_diffusion installation.'
    nuke.message(error_message)
    nuke.tprint(error_message)
