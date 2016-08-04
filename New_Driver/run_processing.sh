for dir in */; do  dir=${dir%*/}; cd $dir; for dir2 in */; do echo $dir2;python tm_match_driver.py tm/tm_c0.jpg $dir2/;done;   cd ..;    echo ${dir##*/}; done

