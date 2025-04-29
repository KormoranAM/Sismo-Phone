# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:20:24 2024

@author: sebas

This script can be decomposed into 2 main Parts : 
    Part 1: extraction of initial times for different sources performed on field 
    Part 2: SVD decomposition of each geophone channel (E,N,Z) in order to compute phase velocity
    of modes QS0 and SH0, and infer Young modulus E and Poisson coefficient nu. 
    
    Extraction of coordinates (f,k) from FK plot of channel Z (flexural waves) is also performed in order to infer 
    ice thickness and ice density in an other Python script : 'Inv_disp_curves_bidir.py'

"""

#%% Import modules 
import numpy as np

import datetime 
import matplotlib
matplotlib.use('TkAgg')  # ou 'Qt5Agg'
import mplcursors
from matplotlib.widgets import RectangleSelector
import matplotlib.pyplot as plt

from obspy.core import UTCDateTime
from obspy import read
import shutil
import os
import re
import tkinter as tk
from tkinter import messagebox
import pickle
from scipy.fftpack import fft, ifft
from scipy.linalg import svd
import warnings
from matplotlib.patches import PathPatch
from scipy.interpolate import make_interp_spline
import warnings
import matplotlib.dates as mdates
from datetime import timezone
from matplotlib.backend_bases import KeyEvent
#%% Set parameters 
year = '2025'
date = '0206' #date format, 'mmdd'
acqu_numb = '0001' #acquisition number 

path2data = date +'/Geophones/'

# set path to geophone correspondence table
geophones_table_path = 'geophones_table'
# channel = 0  # 0 for E, 1 for N, 2 for Z. 

#files need to be organised as: data/0210/Geophones/0001/minised files

geophones_spacing = 3 # space between geophones, in meter 
signal_length = 1 # duration in seconds 
channel_dic = {
    1: "N",
    2: "Z",
    0: "E",}
# ch = channel_dic[channel]

#%% Set plotting parameters 

# Parameters for plots
font_size_medium = 24
font_size_small = 18
plt.rc('font', size=font_size_medium)          # controls default text sizes
plt.rc('axes', titlesize=font_size_medium)     # fontsize of the axes title
plt.rc('axes', labelsize=font_size_medium)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=font_size_small)    # fontsize of the tick labels
plt.rc('ytick', labelsize=font_size_small)    # fontsize of the tick labels   
plt.rc('figure', titlesize=font_size_medium)  # fontsize of the figure title

fig_size = (12,9)
img_quality = 1000 # dpi to save images 

plt.rcParams.update({
    "text.usetex": True}) # use latex

fig_folder = path2data + acqu_numb + '/' + 'Results/' # folder where figures are saved 

if not os.path.isdir(fig_folder):
    os.mkdir(fig_folder)
    


#%% Function definition 

#----------------------------------------------------------------------------------------------
def read_data(path2data):
    """ Read path to folder where data are stored, 
    Extract each stream and create a list of streams 
    Inputs :
        - path2data : path to data files
    Outputs : 
        - streams : list of stream of all geophones 
        - minseed_files : list of files .miniseed """
    miniseed_files = []

    # Iterate over files in the specified directory
    for filename in os.listdir(path2data):
        if filename.endswith(".miniseed"):
            file_path = os.path.join(path2data, filename)
            miniseed_files.append(file_path)

    # Read MiniSEED files
    streams = []
    for file_path in miniseed_files:
        stream = read(file_path)
        streams.append(stream)

    return streams, miniseed_files
#----------------------------------------------------------------------------------------------
def convert_to_utc_times(trace, time_vector):
    """ Create a UTC-time time vector for a given trace 
    Inputs :
        - trace : trace object (obspy) 
        - time_vector : array of time elapsed since beginning of the acquisition 
    Output : 
        - array of UTC time (datetime type)"""
    start_time_utc = UTCDateTime(trace.stats.starttime)
    return [start_time_utc + t for t in time_vector]
#----------------------------------------------------------------------------------------------
def rename_traces(stream, geophones_dict):
    """ Rename geophone traces using geophone_table 
    Inputs : 
        - stream : stream containing traces of each geophone 
        - geophones_dict : dictionnary of geophone correspondance 
    Output : 
        - sorted_stream : stream sorted by name using geophone table of correspondance """
    for trace in stream:
        # Extract the last 5 digits after "SS."
        last_five_digits = trace.stats.station[-5:]

        # Check if the last five digits exist in the geophones dictionary
        if last_five_digits in geophones_dict:
            # Get the corresponding 2-digit number from the geophones dictionary
            new_station_code = geophones_dict[last_five_digits]

            # Replace the last 5 digits in the existing station code with the new ones
            trace.stats.station = f"{trace.stats.station[:-5]}{new_station_code}.{trace.stats.channel}"
        else:
            print(
                f"Warning: No entry found for {last_five_digits} in the geophones table.")

    # Sort traces based on the new station codes
    sorted_stream = sorted(stream, key=lambda trace: trace.stats.station)

    return sorted_stream

#---------------------------------------------------------------------------------------------
def sort_key(trace):
    """ Create a sorting key based on stream stations """
    return trace[0].stats.station

#----------------------------------------------------------------------------------------------


def disconnect_toolbar_events(fig):
    toolbar = fig.canvas.toolbar
    toolbar.toolmanager.remove_tool('zoom')
    toolbar.toolmanager.remove_tool('pan')

# Function to reconnect the toolbar events
def reconnect_toolbar_events(fig):
    toolbar = fig.canvas.toolbar
    toolbar.toolmanager.add_tool('zoom', plt._tools.Zoom)
    toolbar.toolmanager.add_tool('pan', plt._tools.Pan)


#----------------------------------------------------------------------------------------------


def extents(f):
    """ Computes the extents of an array, returns extremities to be used with plt.imshow """
    delta = f[1] - f[0]
    return [f[0] - delta/2, f[-1] + delta/2]


print('fonctions chargées')
###################################################################
#%% -------------------- Loading geophones Data -------------------
###################################################################


# Load geophones table into a dictionary
geophones_dict = {}
with open(geophones_table_path, 'r') as table_file:
    for line in table_file:
        if line.startswith("Num"):
            continue  # Skip header line
        num, geo_sn = line.strip().split()
        # Assuming the last five digits are relevant for comparison
        last_five_digits = geo_sn[-5:]
        geophones_dict[last_five_digits] = num


# Read MiniSEED file directly
seismic_data_streams, miniseed_files = read_data(path2data +'/' +acqu_numb)

# Iterate over streams and rename traces
for i, stream in enumerate(seismic_data_streams):
    seismic_data_streams[i] = rename_traces(stream, geophones_dict)


# Sort the seismic_data_streams based on the custom sorting function
seismic_data_streams = sorted(seismic_data_streams, key=sort_key)

unsorted_seismic_data = seismic_data_streams.copy()
#streams_G11 = unsorted_seismic_data[33:36]
#streams_G12 = unsorted_seismic_data[30:33]
#unsorted_seismic_data[30:33] = streams_G11
#unsorted_seismic_data[33:36] = streams_G12

seismic_data_streams = unsorted_seismic_data

# Find the largest start time (tstart) and the smallest end time (tend) among all streams
tstart = max([stream[0].stats.starttime for stream in seismic_data_streams])
tend = min([stream[-1].stats.endtime for stream in seismic_data_streams])

# Truncate all streams between tstart and tend
for i, stream in enumerate(seismic_data_streams):
    for trace in stream:
        if trace.stats.starttime < tstart:
            trace.trim(tstart, tend, pad=True, fill_value=0)
        elif trace.stats.endtime > tend:
            trace.trim(tstart, tend, pad=True, fill_value=0)


# Extract time_vector and data_vector for the first stream outside the loop
first_stream = seismic_data_streams[0]
first_trace = first_stream[0]
time_array = first_trace.times() # experimental time array 
time_vector = convert_to_utc_times(first_trace, first_trace.times())
datetime_values = [t.datetime.replace(tzinfo=timezone.utc) for t in time_vector]# create datetime UTC array 
data_vector = first_trace.data
start_time_utc = UTCDateTime(first_trace.stats.starttime) 
fs = first_trace.stats.sampling_rate # acquisition frequency (Hz) 

print('data_chargées')

##############################################################################################
#%%----------------------- PLOTTING SELECTED CHANNEL FOR ALL GEOPHONES ------------------ 
##############################################################################################
channel_dict = {'E' : 0, 'N' : 1, 'Z': 2}
composante = input("choisir la composante \nE \nN \nZ \n")
channel = channel_dict[composante]
 #0 for E, 1 for N, 2 for Z. 


#################################################################################
#%%----------------------- SELECTING TIME RANGE FOR PROCESSING ------------------ 
#################################################################################
# Select signal segments manually for each track and each source 
# Store selected times in a dictionnary and save it as a .pkl file 

# open .pkl file and load dictionnary 
filename = 't1_to_time_' + date + '_' + year + '.pkl'
base = path2data
file2save = base + filename
if os.path.isfile(file2save):
    print('Time dictionnary already exists')
    with open(file2save,'rb') as f:
        time_dict = pickle.load(f)

else: 
    time_dict = {'d' + date + 'a' + acqu_numb + 'tS' + '101' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '102' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '103' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '104' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '105' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '106' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '107' + 'N': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '101' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '102' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '103' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '104' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '105' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '106' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '107' + 'E': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '101' + 'Z': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '102' + 'Z': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '103' + 'Z': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '104' + 'Z': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '105' + 'Z': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '106' + 'Z': None,\
            'd' + date + 'a' + acqu_numb + 'tS' + '107' + 'Z': None}



is_zooming_or_panning = False

sample_rate = 10                  

selected_keys = list(time_dict.keys()) 

channel_name = ['E', 'N', 'Z']
selected_keys = [element for element in  selected_keys if element[-1] == channel_name[channel]]

selected_index = 0 
vertical_lines = []
annotations = []
def update_annotations():
    """ Met à jour les annotations sans effacer le reste du graphique """
    global vertical_lines, annotations
    ax = plt.gca()

    # Supprimer les anciennes annotations
    for line in vertical_lines:
        line.remove()
    for text in annotations:
        text.remove()

    vertical_lines.clear()
    annotations.clear()

    # Ajouter les nouvelles annotations
    for key, value in time_dict.items():
        if value != None:
            line = ax.axvline(x=value, color='r', linestyle='--')
            text = ax.text(value, ax.get_ylim()[1] * 0.9, key[-5:], 
                       color='red', verticalalignment='bottom', fontsize=10)
            vertical_lines.append(line)
            annotations.append(text)

    fig.canvas.draw()  # Rafraîchir l'affichage

def on_click(event):
    global selected_index
    global selected_index, is_zooming_or_panning
    if is_zooming_or_panning:
        return  # Ignore la sélection si l'utilisateur est en train de zoomer ou déplacer

    if event.inaxes is not None and event.xdata is not None:  # Vérifie que le clic est dans la zone du graphique et qu'on n'a pas déjà 5 valeurs
        # Stocke la valeur dans le dictionnaire
        print("choisir une source:")

        liste_key_input = []
        for key in selected_keys:
            key_name = key[-5:-1]
            liste_key_input.append(key_name)
            print(key_name)
        key_input = input("Entrez la clé pour cette valeur sélectionnée: ")
        if key_input in liste_key_input:
            time_dict['d' + date + 'a' + acqu_numb + 't' + key_input + channel_name[channel]] = mdates.num2date(event.xdata)
            update_annotations()
        else:
            return



def on_pan_or_zoom(event):
    global is_zooming_or_panning
    is_zooming_or_panning = True 

def on_release(event):
    global is_zooming_or_panning
    is_zooming_or_panning = False


fig, ax = plt.subplots()

selected_indices = range(channel, len(seismic_data_streams), 3)
for k, idx in enumerate(selected_indices):
    print(k)
    tr = seismic_data_streams[idx][0]   # votre objet Trace
    normed = tr.data / np.max(np.abs(tr.data))
    offset = geophones_spacing * (k + 1)
    ax.plot(datetime_values[::10],
            (normed + offset)[::10],
            label=f"Stream {k}")

ax.set_ylabel("Amplitude normalisée + offset")
ax.set_xlabel("Temps")

# --- (3) Classe de gestion du zoom / sélection rectangulaire ---

# --- (6) Affichage ---

ax.set_title(f'Signal superposition for channel {channel}\n Selectioner les sources')
fig.canvas.mpl_connect("button_press_event", on_click)
fig.canvas.mpl_connect("motion_notify_event", on_pan_or_zoom)  
fig.canvas.mpl_connect("button_release_event", on_release)  

plt.ion()
plt.show(block  = True)



print("\n📌 Valeurs sélectionnées :")
time_dict_without_none = {k: v for k, v in time_dict.items() if v is not None}
for key, value in time_dict_without_none.items():
    print(key[-5:], ":", value)
# Save t0 dictionnary in pickle file 
with open(file2save, 'wb') as f:
    pickle.dump(time_dict, f)

print('Time dictionnary saved')
