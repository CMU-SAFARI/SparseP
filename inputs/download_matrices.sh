#!/bin/sh
  
# This script downloads some matrix files
# Download matrix files in mtx format from: https://sparse.tamu.edu/

echo "Starting download..."


if [ ! -f "ldoor.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/GHS_psdef/ldoor.tar.gz
  tar xzf ldoor.tar.gz 
  mv ldoor/ldoor.mtx .
  rm -rf ldoor/
  rm "ldoor.tar.gz"
fi

if [ ! -f "af_shell1.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/Schenk_AFE/af_shell1.tar.gz
  tar xzf af_shell1.tar.gz 
  mv af_shell1/af_shell1.mtx .
  rm -rf af_shell1/
  rm "af_shell1.tar.gz"
fi


if [ ! -f "roadNet-TX.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/SNAP/roadNet-TX.tar.gz
  tar xzf roadNet-TX.tar.gz 
  mv roadNet-TX/roadNet-TX.mtx .
  rm -rf roadNet-TX/
  rm "roadNet-TX.tar.gz"
fi


if [ ! -f "parabolic_fem.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/Wissgott/parabolic_fem.tar.gz
  tar xzf parabolic_fem.tar.gz 
  mv parabolic_fem/parabolic_fem.mtx .
  rm -rf parabolic_fem/
  rm "parabolic_fem.tar.gz"
fi


if [ ! -f "poisson3Db.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/FEMLAB/poisson3Db.tar.gz
  tar xzf poisson3Db.tar.gz 
  mv poisson3Db/poisson3Db.mtx .
  rm -rf poisson3Db/
  rm "poisson3Db.tar.gz"
fi


if [ ! -f "delaunay_n19.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/DIMACS10/delaunay_n19.tar.gz
  tar xzf delaunay_n19.tar.gz 
  mv delaunay_n19/delaunay_n19.mtx .
  rm -rf delaunay_n19/
  rm "delaunay_n19.tar.gz"
fi


if [ ! -f "com-Youtube.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/SNAP/com-Youtube.tar.gz
  tar xzf com-Youtube.tar.gz 
  mv com-Youtube/com-Youtube.mtx .
  rm -rf com-Youtube/
  rm "com-Youtube.tar.gz"
fi


if [ ! -f "pkustk14.mtx" ]; then
  wget -N https://suitesparse-collection-website.herokuapp.com/MM/Chen/pkustk14.tar.gz
  tar xzf pkustk14.tar.gz 
  mv pkustk14/pkustk14.mtx .
  rm -rf pkustk14/
  rm "pkustk14.tar.gz"
fi


echo "Finished downloading and extracting .mtx files"
