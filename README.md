# GPU
## Run Commands
nvcc -o bubble_sort_comparison bubble_sort_comparison.cu <br/>
./bubble_sort_comparison <number_of_elements> <threads_per_block> <number_of_blocks> <br/>
<br/>
nvcc -o bitonic_sort_comparison bitonic_sort_comparison.cu <br/>
./bitonic_sort_comparison <number_of_elements> <threads_per_block> <number_of_blocks> <br/>
<br/>
nvcc -o sample_sort_comparison sample_sort_comparison.cu <br/>
./sample_sort_comparison <number_of_elements> <br/>
<br/>
nvcc -o odd_even_transposition_sort odd_even_transposition_sort.cu <br/>
./odd_even_transposition_sort <number_of_elements> <threads_per_block> <number_of_blocks> <br/>
