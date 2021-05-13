export GERL=/data/yunfanhu/GERL_V2
mkdir data outputs outputs/L outputs/L/base@examples/
cd data
mkdir L L/train L/dev L/test L/examples vocabs 
cd ..


python -m src.scripts.build_train_samples
python -m src.scripts.build_vocabs
python -m src.scripts.build_neighbors
python -m src.scripts.build_neighbors --type=test
python -m src.scripts.build_training_examples
python -m src.scripts.build_eval_examples
python -m src.scripts.build_test_examples
python -m src.scripts.split_data
python -m src.scripts.split_data -i L/examples/test_examples.tsv -o L/examples/test_examples
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.train
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m src.test training.validate_epoch=6
python -m src.gen_submission 