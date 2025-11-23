# python -m src.runners.run --dataset_name wikidata --model BUDDY
# python -m src.runners.run --dataset_name twitter --model BUDDY
# python -m src.runners.run --dataset_name ppi --model BUDDY
# python -m src.runners.run --dataset_name dblp --model BUDDY
# python -m src.runners.run --dataset_name blogcatalog --model BUDDY

python -m src.runners.run --dataset_name wikidata1k_multiclass --model BUDDY --multiclass
# python -m src.runners.run --dataset_name wikidata5k_multiclass --model BUDDY --multiclass
# python -m src.runners.run --dataset_name wikidata10k_multiclass --model BUDDY --multiclass