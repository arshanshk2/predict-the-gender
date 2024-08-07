Gender Prediction from Voice Samples
Project Overview
Ever wondered if a computer can tell if a voice belongs to a male or female? This project explores exactly that! We'll build a model to predict gender based on different acoustic features of voice recordings.

What's This About?
We've got a dataset with 3,168 voice samples from both men and women. These samples have been analyzed using some cool R packages (seewave and tuneR), giving us a bunch of features to work with.

The Challenge
Create a model that can accurately classify whether a voice is male or female using the provided acoustic properties.

The Data
The dataset includes the following features for each voice sample:

meanfreq: Average frequency (in kHz)
sd: Standard deviation of frequency
median: Median frequency (in kHz)
Q25: First quantile (in kHz)
Q75: Third quantile (in kHz)
IQR: Interquantile range (in kHz)
skew: Skewness of the frequency distribution
kurt: Kurtosis of the frequency distribution
sp.ent: Spectral entropy
sfm: Spectral flatness
mode: Mode frequency
centroid: Frequency centroid
peakf: Peak frequency (frequency with the highest energy)
meanfun: Average fundamental frequency
minfun: Minimum fundamental frequency
maxfun: Maximum fundamental frequency
meandom: Average dominant frequency
mindom: Minimum dominant frequency
maxdom: Maximum dominant frequency
dfrange: Range of the dominant frequency
modindx: Modulation index
label: Gender (male or female)

How We’ll Do It
Clean the Data: Get rid of any missing values.
Visualize: Show the gender distribution with a pie chart.
Split the Data: Use 80% for training and 20% for testing.
Train Models: We'll try out several algorithms:
Decision Tree
Random Forest
K-Nearest Neighbors (KNN)
Logistic Regression
Support Vector Machine (SVM)
Evaluate: Check the performance of each model using confusion matrices and classification reports.
Pick the Best: Identify which model does the best job.
Why This is Cool
Accuracy: Find out which model is best at predicting gender from voice features.
Comparison: See how different algorithms stack up against each other.
Insight: Learn which acoustic features are most important for gender prediction.
Join me on this journey to see if machines can successfully determine gender from voice alone!
