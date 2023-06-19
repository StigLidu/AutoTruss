python Stage1/continuous_uct_3d.py --config $1 --run-id $2
python Stage1/noise_input_permutation_format_transfer.py --config $1 --run-id $2
python Stage2/main_3d.py --config $1 --run-id $2 --EmbeddingBackbone Transformer
python apps/draw.py --config $1 --run-id $2 --draw-file PostResults/$1/$2/_best.txt