
prefix="docker run --rm -it -v ./:/home/worker/work --gpus all meetingdocker/ml:ss python -m src.runners.run"
# prefix="python -m src.runners.run"

# $prefix --dataset_name wikidata --model BUDDY
# $prefix --dataset_name twitter --model BUDDY
# $prefix --dataset_name ppi --model BUDDY
# $prefix --dataset_name dblp --model BUDDY
# $prefix --dataset_name blogcatalog --model BUDDY

$prefix --dataset_name wikidata1k_multiclass --model BUDDY --multiclass
# $prefix --dataset_name wikidata5k_multiclass --model BUDDY --multiclass
# $prefix --dataset_name wikidata10k_multiclass --model BUDDY --multiclass