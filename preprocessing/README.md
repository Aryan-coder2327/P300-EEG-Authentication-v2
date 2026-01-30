\## EEG Preprocessing Module



This module handles EEG data preprocessing for the P300-based

authentication system.



\### Steps Implemented

1\. Loading EEG data from BNCI Visual P300 dataset

2\. EEG channel identification

3\. Bandpass filtering (0.1–30 Hz)

4\. Flash onset detection

5\. Epoch extraction (-200 ms to +800 ms)

6\. Baseline correction

7\. Averaged P300 ERP computation



The output of this module is used by subsequent authentication

and machine learning components.



