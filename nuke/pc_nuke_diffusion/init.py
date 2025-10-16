import os
import sys
current_directory = os.path.dirname(os.path.abspath(__file__))
nuke.pluginAppendPath(
    os.path.join(current_directory, 'plugin_libs'))

print(nuke.NUKE_VERSION_MAJOR, nuke.NUKE_VERSION_MINOR)

if nuke.NUKE_VERSION_MAJOR==15 and nuke.NUKE_VERSION_MINOR==1:
    nuke.pluginAppendPath(
        os.path.join(current_directory, 'plugin_libs/Release_15.1'))

if nuke.NUKE_VERSION_MAJOR==15 and nuke.NUKE_VERSION_MINOR==2:
    nuke.pluginAppendPath(
        os.path.join(current_directory, 'plugin_libs/Release_15.2'))

try:
    nuke.load('pc_nuke_diffusion')
except RuntimeError:
    error_message = 'Error! Failed to load pc_nuke_diffusion library. Please check pc_nuke_diffusion installation.'
    print(error_message)
    #nuke.tprint(error_message)
