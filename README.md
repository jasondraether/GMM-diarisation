# SuperMega-Classifier
A work in progress. Utilizing Gaussian Mixture Models, we seek to identify the current speaker at a given moment in the podcast. The SuperMega Podcast has two speakers, Matt and Ryan. Using MFCC's and MFCC deltas with a selected dataset from the two speakers, the GMM computes a log-likelihood of an unseen MFCC from being within a speaker's GMM cluster.
