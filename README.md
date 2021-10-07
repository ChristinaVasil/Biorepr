# Biorepr

This project aims to extract multi-omic data from Genomic Data Commons Database, construct feature vectors and personalized graphs in order to successfully predict the tumor stage and case/control status, through Machine Learning.

## Package Installation and data download

Download the Data Transfer Tool `gdc-client` from Genomic data commons [here](https://gdc.cancer.gov/access-data/gdc-data-transfer-tool) and unzip it:
```
wget https://gdc.cancer.gov/files/public/file/gdc-client_v1.6.1_Ubuntu_x64.zip
unzip gdc-client_v1.5.0_Ubuntu_x64.zip`
```

After the selection of the appropriate files in GDC, the user should have downloaded a  `gdc_manifest.*.txt` file.

Download the -omic data using the Data Transfer Tool:

`gdc-client download -m gdc_manifest.*.txt`

The command above will download all the selected files from GDC, and store them in folders for each sample. Since each file is compressed, you need to decrompress all files, across all folders:

`gzip -r ./`

## Kidney Cancer

Create a file which matches status with unique sample ids:
```
cut -f 7,8 gdc_sample_sheet.2021-06-16.tsv > status.txt
cat status.txt  | sort | uniq > status_uniq.txt
```

The `status_uniq.txt` file is used in our perl script in order to add an extra `Status` column in the final feature vector.

You now run the perl scipt which will output a feature vector .txt file:

`perl gdcscript.pl`

The feature vector is used in our python script, which:

  1. Makes sure that the feature vector is valid
  2. Replaces all types of N/A values with `nan`
  3. Perfoms normalization and filtering across all multi-omic values
  4. Adds a new target feature Tumor Stage, using a clinical file from GDC
  5. Constructs generalized and personalised graphs per sample
  6. Performs classification on Case status and Tumor Stage.
  
Prior to running the python script, the clinical file requires some cleaning of incosistencies like "stage i", "Stage i", "stage I". This was performed mostly manually (To Do: Add a python process or a bash script).
```
sed -i 's/Stage/stage/g' clinical.tsv
sed -i 's/stage I/stage i/g' clinical.tsv
```

Now you may run the python script like this:

`python3 preProcessing.py`
