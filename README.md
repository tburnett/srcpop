# Fermi gamma-ray source population study
These notebooks use spectral and time-dependence characteristics of the 4FGL-DR4 point source catalog contents as input to a ML training of the associated members to predict the identities of the unassociated ones.

Code dependencies beside usual Jupyter:
* [wtlike](https://github.com/tburnett/wtlike), specifically a few modules in its utilities folder
* seaborn

Data:
* `files/fermi_sources_v2.csv`


Notebooks:
* [Perform machine learning classification](machine_learning.ipynb)
* [Spectral parameters and UW/DR4 difference](study_spectra.ipynb).
* [Compare UW and 4FGL Epeaks](compare_epeak.ipynb)
* [Compare the *predicted*  pulsars with the *identified* set](pulsar_pop.ipynb)
* [Study curvature](curvature.ipynb)

