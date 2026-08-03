import numpy as np
import pandas as pd

# The MEMA paper says: 32 electrodes, 10-10 system, ref=CPz, ground=FPz
# From the paper's topographic maps, the standard order for ZhenTec-NT1-32 is:
# (This is the standard 10-10 layout used in most Chinese EEG studies with this device)

# Standard ZhenTec NT1-32 channel order (30 EEG + 2 EOG):
CHANNEL_NAMES_32 = [
    'FP1', 'FP2',  'F7',  'F3',  'Fz',  'F4',  'F8',  'FT7',
    'FC3', 'FCz',  'FC4', 'FT8', 'T7',  'C3',  'Cz',  'C4',
    'T8',  'TP7',  'CP3', 'CPz', 'CP4', 'TP8', 'P7',  'P3',
    'Pz',  'P4',   'P8',  'O1',  'Oz',  'O2',  'HEOG','VEOG'
]

# But we have 32 columns and column 17 = 187500 (constant = reference voltage?)
# And columns 30-31 have huge values (gyroscope-like)
# Let's verify: in the MEMA paper, ref=CPz=channel 19 in 10-10
# But col 17 is constant... that doesn't match CPz position in the standard order

# Let me try a different standard order. Some ZhenTec systems use:
# The device manual order might differ from the 10-10 alphabetical order.
# 
# From other XJTU-EEG papers, the channel order often is:
# FP1, FP2, F3, Fz, F4, F7, F8, FC3, FCz, FC4, FT7, FT8,
# C3, Cz, C4, T7, T8, CPz(ref), CP3, CP4, TP7, TP8,
# P3, Pz, P4, P7, P8, O1, Oz, O2, HEOG, VEOG
#
# In this order, col 17 = CPz (reference) which makes sense as constant!

CHANNEL_NAMES_32_V2 = [
    'FP1', 'FP2', 'F3',  'Fz',  'F4',  'F7',  'F8',  'FC3',
    'FCz', 'FC4', 'FT7', 'FT8', 'C3',  'Cz',  'C4',  'T7',
    'T8',  'CPz', 'CP3', 'CP4', 'TP7', 'TP8', 'P3',  'Pz',
    'P4',  'P7',  'P8',  'O1',  'Oz',  'O2',  'HEOG','VEOG'
]

print("Channel mapping V2 (col 17 = CPz reference):")
for i, name in enumerate(CHANNEL_NAMES_32_V2):
    note = ""
    if i == 17: note = " <-- CONSTANT 187500 (reference)"
    if i >= 30: note = " <-- likely non-EEG (large values)"
    print(f"  col {i:2d}: {name:4s}{note}")

# For the 7-channel subset (Paper 1/3): F3, F4, Fz, C3, C4, Cz, Pz
# In V2 mapping: F3=col2, F4=col4, Fz=col3, C3=col12, C4=col14, Cz=col13, Pz=col23
target_channels = ['F3', 'F4', 'Fz', 'C3', 'C4', 'Cz', 'Pz']
target_indices = [CHANNEL_NAMES_32_V2.index(ch) for ch in target_channels]
print(f"\n7-channel subset indices: {dict(zip(target_channels, target_indices))}")

# Verify these columns look like EEG (not EOG/reference)
txt_path = r'MEMA Dataset\MEMA\For_graph\Subject1\Subject1_a1.txt'
df = pd.read_csv(txt_path, header=None, nrows=500, usecols=target_indices)
print(f"\n7-channel data sample stats:")
for col_idx, ch_name in zip(target_indices, target_channels):
    vals = df[col_idx]
    print(f"  {ch_name} (col {col_idx}): mean={vals.mean():.1f}, std={vals.std():.1f}, range=[{vals.min():.1f}, {vals.max():.1f}]")
