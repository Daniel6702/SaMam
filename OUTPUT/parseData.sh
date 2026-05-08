#!/bin/bash
 
# Output CSV file
OUTPUT_CSV="results.csv"
 
# Write CSV header
echo "folder,ArtFID,FID,LPIPS,CFSD" > "$OUTPUT_CSV"
 
# Loop through all subdirectories in OUTPUT/
for dir in $(ls -t -d */); do
    eval_file="${dir}evaluation.txt"
 
    # Skip if evaluation.txt doesn't exist
    [ -f "$eval_file" ] || continue
 
    folder=$(basename "$dir")
 
    # Read metric values from evaluation.txt
    # Line 1: ArtFID, FID, LPIPS, LPIPS_gray
    # Line 2: CFSD
    artfid=$(grep -oP 'ArtFID:\s*\K[\d.]+' "$eval_file" | awk '{printf "%.3f", $1}')
    fid=$(grep -oP '(?<![A-Za-z])FID:\s*\K[\d.]+' "$eval_file" | awk '{printf "%.3f", $1}')
    lpips=$(grep -oP '(?<![A-Za-z])LPIPS:\s*\K[\d.]+' "$eval_file" | head -1 | awk '{printf "%.3f", $1}')
    cfsd=$(grep -oP 'CFSD:\s*\K[\d.]+' "$eval_file" | awk '{printf "%.3f", $1}')

    echo "${folder},${artfid},${fid},${lpips},${cfsd}" >> "$OUTPUT_CSV"
done
 
echo "Done. Results written to $OUTPUT_CSV"

