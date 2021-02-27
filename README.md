﻿# Alexa Marketplace Analysis

This repository uses the compilation of json and csv files of Jide to create a unique dataset to performe the anlaysis on Alexa skills ecosystem for the paper <b>submitted 1/8/2020</b> at <b>NDSS'21 Fall</b>.
Jide collected the skills ecosystem half a year ago and a few weeks ago, in two different datasets per market (namely old and new). Some of the original files are not included in the repo, since they were huge and had many repetitions. 

This repo contains:
<ul>
        <li><b>_AnalysePhonetics/      </b>: Folder that includes used to estimate phonetic distance between skill names using CMU dictionary and normalised Levehnstein distance. Inside the folder you can find the jupyter notebook used to perfom the experiments, and a set of jsons used to speed up the phonetic analysis.</li> 
        <li><b>NewFullDataset/         </b>: Folder that contains the newly created .json datasets that combines both skills json and skill linkage</li>
        <li><b>out/                    </b>: Output folder for the .json comparison analysis between old and new skill crawling</li>
        <li><b>skills/                 </b>: Original json and excel files with the old and new crawling and skill linkage. Note crawling is not complete in some cases, as Jide's files were huge. See paper for original datasets. Also, the dataset contains repetitions and some skills and not correctly parsed. Each line in the file corresponds with a json object</li>
        <li><b>CreateNewFullDataset-MapTraceabilityToNewSkills.ipynb</b>: Jupyter notebook used to create the new dataset saved in NewFullDataset</li>
        <li><b>FiguresDataset.ipynb    </b>: Jupyter notebook with different figures and analyses used in the paper</li>
        <li><b>Comparedatasets.ipynb   </b>: Jupyter notebook that compares the previous skill dataset with the new, and creates the files in out/</li>
</ul>



## Main analysis, FiguresDataset.ipynb
Jupyter notebook <b>FiguresDataset.ipynb</b> in which we collect all data used in the different figures and tables of the paper. Here we use the <b>NewFullDataset/</b> created with <b>CreateNewFullDataset-MapTraceabilityToNewSkills.ipynb</b>. This way we can follow, check and see if there are any mistakes in the code or collectiuon of the data. Also, we can duplicate the figures. Although the sections do not generate the paper figrues (these are generated using Excel), the information presented in the excel figures is obtained from the results of this jupyter notebook.

## Phonetic analisys, _AnalysePhonetics/ folder
All the necessary information and libraries to estimate the phonetic simliarity between skills are included in this folder. 
warning, computing the Levehnstein phonetic distance between all skills is very slow, so some tricks were used.


## Comparison between snapshots, Comparedatasets.ipynb
<b> Notice this analysis and output files do not include traceability</b>
We compared all data regarding the different snapshots per market. For every market, we perform the next analysis:
<ul>
        <li>newSkills:  Skills that did not exist in previous snapshot of the market and that have been created since then.</li>
        <li>removedSkills: Skills that were removed since the last snapshot</li>
        <li>stillExistingSkills: Skills that exist in the old snapshot AND in the new</li>
        <li>changedPermissionsSkills: skills that exist in the previous snapshot and in the new AND that changed permissions (added or removed)</li>
</ul>
The jsons resulting from this comparison can be found in the <b>Out/</b> folder. 
The jupyter notebook performing the analysis and results is <b>Comparedatasets.ipynb</b>



