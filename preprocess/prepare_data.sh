python get_amr_dict.py --dataset_name=Causal-TimeBank
python get_amr_triple.py --dataset_name=Causal-TimeBank
python get_align_data.py --dataset_name=Causal-TimeBank

python split_topic.py
python build_graph_cv.py
