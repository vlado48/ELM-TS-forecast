# ELM-TS-forecast
I found it interesting to try to apply Extreme learning machine for time series forecast.
What are ELM?

ELM:

<img src="https://petlew.com/assets/img/slfn.jpg" width=50% height=50%>

Single hidden Layer Feedforward Network (SLFNs)
- Hβ = T ; where H is hidden layer matrix, β is output weigths matrix, T is vector of labels
- Weigths are randomly assigned, thus H known
- solve for Hβ - T = 0 for β by Moore-Penrose inverse
<pre>
DataPrep.ipynb   --- Data exploration, filling, cleaning, feature engineering

M1_ELM.ipynb     --- ELM implementation for price predicion and directionality prediction

Pipeline.ipynb   --- Example of higher level functions used directly on dataset

codebase.py      --- Cleaned and wrapped function

ELM.py           --- ELM implementation (source: https://github.com/burnpiro/elm-pure)

Dataset.csv      --- Original dateset

Dataset_tidy.csv --- Postprocessed dataset
</pre>

