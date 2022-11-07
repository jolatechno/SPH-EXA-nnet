#!/bin/bash

file_types=".cpp .inl .h .hpp .cu .cuh"

for extension in ${file_types}; do
	echo "clang formating ${extension} files:"

	files=$(find . -type f -name "*${extension}" -not -path "*/build/*")
	for file in ${files}; do
		echo -e "\tclang-format -i --style=file ${file}"
		clang-format -i --style=file ${file}
	done

	echo ""
done