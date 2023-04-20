#!/bin/bash

echo "Run handin 3" 

echo "Creating plotting directory if it doesn't already exist"
if [ ! -d "plots" ]; then
  echo "Directory does not exist, create it!" 
  mkdir plots
fi

echo "Download halo data"
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m11.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m12.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m13.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m14.txt
wget https://home.strw.leidenuniv.nl/~daalen/Handin_files/satgals_m15.txt

echo "Run the first script"
python3 NUR_handin3a.py > NUR_handin3a.txt

echo "Run the second script"
python3 NUR_handin3b.py > NUR_handin3b.txt

echo "Run the third script"
python3 NUR_handin3c.py > NUR_handin3c.txt

echo "Generate the pdf"
pdflatex main.tex
bibtex main.aux
pdflatex main.tex
pdflatex main.tex
