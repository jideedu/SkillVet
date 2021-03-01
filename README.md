# Alexa Marketplace Analysis

This repository uses the compilation of json and csv files collected from Amazon website to create a unique dataset to perform the traceability anlaysis on Alexa skills ecosystem.
We collected the skills ecosystem few months ago. The original files are included in the skills.zip file.

This repo contains:
<ul>
        <li><b>_AnalysePhonetics/      </b>: Folder that includes used to estimate phonetic distance between skill names using CMU dictionary and normalised Levehnstein distance. Inside the folder you can find the jupyter notebook used to perfom the experiments, and a set of jsons used to speed up the phonetic analysis.</li>
        <li><b>NewFullDataset/         </b>: Folder that contains the newly created .json datasets that combines both skills json and skill linkage for faster computing</li>
        <li><b>skills.zip                 </b>: Original json files of the crawl data from Amazon Alexa skills website. The dataset contains repetitions. Each line in the file corresponds with a json object</li>
        <li><b>CreateNewFullDataset-MapTraceabilityToNewSkills.ipynb</b>: Jupyter notebook used to create the new dataset saved in NewFullDataset</li>
        <li><b>FiguresDataset.ipynb    </b>: Jupyter notebook with different figures and analyses used in the paper</li>
        <li><b>Comparedatasets.ipynb   </b>: Jupyter notebook that can be used to compare skill dataset with the any new files.</li>
        <li><b>Traceability Folder   </b>: This folder consist of file use in training the traceability model and the datasets used for the validation.</li>

</ul>



## Main analysis, FiguresDataset.ipynb
Jupyter notebook <b>FiguresDataset.ipynb</b> in which we collect all data used in the different figures and tables of the paper. Here we use the <b>NewFullDataset/</b> created with <b>CreateNewFullDataset-MapTraceabilityToNewSkills.ipynb</b>. This way we can follow, check and see if there are any mistakes in the code or collectiuon of the data. Also, we can duplicate the figures. Although the sections do not generate the paper figrues (these are generated using Excel), the information presented in the excel figures is obtained from the results of this jupyter notebook.

## Phonetic analisys, _AnalysePhonetics/ folder
All the necessary information and libraries to estimate the phonetic simliarity between skills are included in this folder.
warning, computing the Levehnstein phonetic distance between all skills is very slow, so some tricks were used.
