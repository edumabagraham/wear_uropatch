This directory contains the plots from retraining the 5s_0.5 data test sets with RandomForest.
1. The average of the numerical hyperparameters were used. The mode for the categorical hyperparameters were used.
2. The predictions of the model was then used to create the overlays on the original plot.
3. For the overlays to be possible, we consider the start-time and end-time used when the features were extracted. This is because one prediction coresponds to a window so we have to know where the window begins and when it end to be able to span the effect of the prediction on the plot. 
The start-time and end-time are already stored where extracting features, the center-time as well. 