# GMM-diarisation
Utilizing Gaussian Mixture Models, we seek to identify the current speaker at a given moment in an audio file. Each speaker has a GMM trained on a dataset of the speaker's Mel Frequency Cepstral Coefficients, as well as 1st and 2nd order 'Deltas' of these MFCCs. For classifying a new, unlabeled sample, the prediction is done by selecting the GMM (out of the list of speaker GMM's) that produces the highest log-likelihood on the given sample.

This project, for now, is on the backburner, as it seems the accuracy of the task is limited by the effectiveness of GMMs and the distribution of the data (if you'd like to see why this doesn't really work well on MFCCs, run the method in plot.py for a set of MFCCs. At least for my two speakers, the distributions are pretty close in the scatterplot matrix, making them hard for the GMM to separate). Here's mine:

![MFCC Plot](https://github.com/jasondraether/GMM-diarisation/edit/master/demo.png)

This model is setup for Bayesian hyperparameter tuning, which I found was really interesting and worked out way better than grid search (see tune.py).

You can view run.py for an example on how to use the modules, it should be compatible with your own audio data provided you keep the same metadata for each channel and make the correct function calls.
