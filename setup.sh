#!/bin/bash
echo Setting up CS231A Problem Set Repository...
echo ...
echo ...
pset=${PWD##*/} # assign current directory name to variable

# check if pset is valid
re='^[0-5]+$'
if ! [[ ${pset: -1} =~ $re ]] ; then
	echo "Problem Set '$pset' does not exist" >&2; exit 1
fi

url="https://web.stanford.edu/class/cs231a/hw/"
pyt_end="_code.zip"
pdf_end=".pdf"
tex_end="_template.zip"

pyt_zip=$pset$pyt_end
tex_zip=$pset$tex_end

pyt_wget=$url$pyt_zip
pdf_wget=$url$pset$pdf_end
tex_wget=$url$tex_zip

echo ------------------------------------------------------
echo Downloading PDF for Problem Set ${pset: -1}...
wget $pdf_wget
echo ------------------------------------------------------
echo Downloading Python scripts and data...
cd data; wget $pyt_wget;
echo Unzipping Python scripts and data...
unzip $pyt_zip; rm *.zip
echo Moving Python scripts to ./scripts/ folder...
mv *.py ../scripts/; cd ..
echo ------------------------------------------------------
echo Downloading LaTeX files...
cd docs; wget $tex_wget;
echo Unzipping LaTeX files in ./docs/ folder...
unzip $tex_zip; rm *.zip
cd ..
echo ------------------------------------------------------
echo All files successfully downloaded for Problem Set ${pset: -1}.

