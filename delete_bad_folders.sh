#!/bin/bash

cd models

for folder in *
do
	cd $folder
	shopt -s nullglob
	numfiles=(*)
	numfiles=${#numfiles[@]}

	if (( $numfiles < 2 ))
	then
		cd ..
		rm -rf $folder
	else
		cd ..
	fi

done

cd ..
