export GERL=/path/to/GERL
cd data
mkdir train dev test vocabs

python -m src.scripts.build_train_samples.py
python -m src.scripts.build_vocabs.py
python -m src.scripts.build_neighbors.py
python -m src.scripts.build_neighbors.py --type=test
python -m src.scripts.build_train_examples.py
python -m src.scripts.build_eval_examples.py
python -m src.scripts.build_test_examples.py
python -m src.scripts.split_data.py 
python -m src.scripts.split_data.py -i L/examples/test_examples.tsv -o L/examples/test_examples
CUDA_VISIBLE_DEVICES=0,1,2,3 python src.train.py 
CUDA_VISIBLE_DEVICES=0,1,2,3 python src.test.py training.validate_epoch=6  