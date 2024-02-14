mkdir -p data
cd data/

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2010/VOCtrainval_03-May-2010.tar
tar -xvf VOCtrainval_03-May-2010.tar
mv VOCdevkit/ PASCAL-Part

cd PASCAL-Part
wget https://roozbehm.info/pascal-parts/trainval.tar.gz
tar -zxvf trainval.tar.gz

cd ..
rm VOCtrainval_03-May-2010.tar
rm PASCAL-Part/trainval.tar.gz
