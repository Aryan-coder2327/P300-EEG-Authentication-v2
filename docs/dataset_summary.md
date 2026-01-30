\# Dataset Summary



\## Dataset Name

BNCI Horizon – Visual P300 Speller Dataset (Guger et al., 2009)



\## Description

This dataset contains EEG recordings from 10 subjects

performing a visual P300 spelling task using a

row/column oddball paradigm. Rare target flashes

elicit a P300 event-related potential.



\## Data Format

\- MATLAB (.mat) files

\- Each subject file contains:

&nbsp; - train run

&nbsp; - test run



\## Key Properties

\- Number of subjects: 10

\- EEG channels: 8 (plus stimulus channels)

\- Paradigm: Visual row/column P300 speller

\- Stimulus classes: Target / Non-target

\- ERP of interest: P300 (~300 ms post-stimulus)



\## Triggering Information

\- Channel 10: Flash ID (1–12)

\- Channel 11: Target label (1 = target, 0 = non-target)



\## Intended Use in Project

The dataset will be used to:

\- visualize raw EEG signals

\- implement EEG preprocessing

\- extract stimulus-locked epochs

\- generate averaged P300 waveforms



