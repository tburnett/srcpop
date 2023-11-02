# Fermi gamma-ray source population study
These notebooks use spectral and time-dependence characteristics of the 4FGL-DR4 point source catalog contents as input to a ML training of the associated members to predict the identities of the unassociated ones.

Notebooks contain further analysis of the characteristics of the unassociated ones.

Code dependencies
* Jupyter, etc.
* seaborn

Files in `files` subfolder:
* `fermi_sources_v2.csv`<br>
This contains a list of ts>25 uw1410 sources with Baysean Block info.
*  


Notebooks:
* [Perform machine learning classification](machine_learning.ipynb)
* [Spectral parameters and UW/DR4 difference](study_spectra.ipynb).
* [Compare UW and 4FGL Epeaks](compare_epeak.ipynb)
* [Compare the *predicted*  pulsars with the *identified* set](pulsar_pop.ipynb)
* [Study curvature](curvature.ipynb)
* [Paper musings](paper.ipynb)
* [Unid analysis](unid-dr4.ipynb)
