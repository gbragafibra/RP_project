#!/bin/bash


output="test_results.txt"

#Create the file

if [ ! -e "$output" ]; then
	touch "$output"
fi



#Clear the file beforehand
> $output


echo "Running!"

python3 main.py > "$output" 2>&1

echo "Finished!"

# Also need to add selection
# for the best model with the
# lowest val error
# and then have it run that
# command
