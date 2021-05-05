# Stroke-Data-Analysis-CPSC322
Classification of a Stroke dataset based on given lifestyle values. Goal to be create a classification accurate enough to predict if an individual is at risk for having stroke.

## The Project
Mine a dataset in order to produce classification results that can accurately predict outcomes.  

Out dataset has attributes detailing regular aspects of an individuals life, and whether or not they had a stroke giving those aspects. We have classified and mined this dataset in order to be able to predict IF someone is at risk for having a stroke in their lifetime.

## How to Run
Our project is run out of the docker container home. In order to run it, you will need to install docker as well as the continuumio/anaconda3:2020.11 container image to go with it.
Visual Studio code, as well as Jupyter lab are both well known platforms that run this program well. You should be able to git clone this repository if you are interested in our classification results, or you can head to our heroku site where it is hosted and play with the implementation.

## Project Organization
**sklearn:** our classifiers that we used are kNN classifier, decision tree classifier, and random forest classifier coded by hand but all based upon sklearns implementations. The sklearn folder holds the neccessary files represented by this library.  
**EDA:** the EDA jupyter notebook is our storybook. It walks through every stage of the dataset and what we did to clean it, classify it, organize it and use it!

