#!/bin/bash

# Initial dimension to test
start_dim=974
# Maximum dimension to test
max_dim=1032
# Step size to increase dimensions
step=2

# File to store output logs
log_file="dimension_test_results.log"

# Clear previous logs
echo "" > $log_file

# Loop over dimensions
for (( dim=$start_dim; dim<=$max_dim; dim+=$step ))
do
    echo "Testing with dimension: $dim" | tee -a $log_file
    # Run the training script with the current dimension
    torchrun --standalone --nproc_per_node=4 train.py --dim=$dim >> $log_file 2>&1
    
    # Check the exit code of the script
    if [ $? -eq 0 ]; then
        echo "Success with dimension: $dim" | tee -a $log_file
        break
    else
        echo "Failed with dimension: $dim" | tee -a $log_file
    fi
done

echo "Testing completed. Check $log_file for results."
