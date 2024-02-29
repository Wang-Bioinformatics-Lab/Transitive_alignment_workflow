import numpy as np
import collections
from typing import List, Tuple
import csv
import math
import argparse
import pickle
import pandas as pd


SpectrumTuple = collections.namedtuple(
    "SpectrumTuple", ["precursor_mz", "precursor_charge", "mz", "intensity"]
)

def norm_intensity(intensity):
    return np.copy(intensity) / np.linalg.norm(intensity)
    # return(intensity)
def read_mgf(filename):
    spectra = []  # List to store all spectra
    with open(filename, 'r') as file:
        spectrum = None  # Current spectrum being processed
        for line in file:
            line = line.strip()
            if line == 'BEGIN IONS':
                spectrum = {'peaks': [],'m/z array':[],'intensity array':[]}  # Initialize a new spectrum
            elif line == 'END IONS':
                spectra.append(spectrum)  # Add the completed spectrum to the list
                spectrum = None  # Reset for the next spectrum
            elif spectrum is not None:
                if '=' in line:  # Property line
                    key, value = line.split('=')
                    spectrum[key.lower()] = value  # Convert key to lowercase for consistency
                else:  # Peak line
                    mz, intensity = line.split()
                    spectrum['peaks'].append((float(mz), float(intensity)))
                    spectrum['m/z array'].append(float(mz))
                    spectrum['intensity array'].append(float(intensity))
    return spectra
def mgf_processing(spectra):
    spec_dic = {}
    for spectrum in spectra:
        mz_array = spectrum['m/z array']
        intensity_array = spectrum['intensity array']
        filtered_mz = []
        filtered_intensities = []
        precursor_value = float(spectrum['pepmass'])
        charge = int(spectrum['charge'].rstrip('+'))
        scans = int(spectrum['scans'])
        for i, mz in enumerate(mz_array):
            peak_range = [j for j in range(len(mz_array)) if abs(mz_array[j] - mz) <= 25]
            sorted_range = sorted(peak_range, key=lambda j: intensity_array[j], reverse=True)
            if i in sorted_range[:6]:
                if abs(mz - precursor_value) > 17:
                    filtered_mz.append(mz)
                    filtered_intensities.append(intensity_array[i])
        filtered_intensities = [math.sqrt(x) for x in filtered_intensities]
        spec_dic[scans] = SpectrumTuple(precursor_value, charge, filtered_mz,
                                                       norm_intensity(filtered_intensities))
    return spec_dic
if __name__ == '__main__':
    # pass arguments
    parser = argparse.ArgumentParser(description='Using realignment method to reconstruct the network')
    parser.add_argument('-c', type=str, required=True, default="specs_ms.mgf", help='mgf filename')

    args = parser.parse_args()
    mgf_filename = args.c

    print("start build the spectrum dictionary")
    # creat the spectrum dictionary
    spectra = read_mgf(mgf_filename)
    spec_dic = mgf_processing(spectra)
    with open('spec_dic.pkl', 'wb') as output_file:
        pickle.dump(spec_dic, output_file)
    print("finish write the spectrum dictionary into spec_dic.pkl")

