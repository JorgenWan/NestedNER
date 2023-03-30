#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

data_name=ace04
archi_name=nner_slg

home_dir=/NAS2020/Workspaces/NLPGroup/juncw
code_dir=$home_dir/codebase/NER/nested_NER/nested_ner
data_dir=$home_dir/database/NER/nested_ner/$data_name
old_data_dir=$home_dir/database/NER/nested_ner/$data_name/old_data

model_dir=$home_dir/database/NER/nested_ner/__models/$data_name/$archi_name
result_dir=$home_dir/database/NER/nested_ner/__results/$data_name/$archi_name

mkdir -p $model_dir
mkdir -p $result_dir

shell_id=41_18

ulimit -n 40000
export LOGLEVEL=INFO

export LOGFILENAME=$result_dir/"$shell_id".log

export LOGFILENAME=$result_dir/"$shell_id".log
python -W ignore $code_dir/train.py \
    --data_dir $data_dir --archi "$archi_name"_"$data_name"  \
    --seed 1 --max_sentences 2 --eval_max_sentences 1 \
    --lr_scheduler linear_with_warmup --warmup_steps 0.01 --max_epoch 50 \
    --optimizer adamw --lr 2e-5 --other_lr 1e-3 --weight_decay 1e-10  \
    --update_freq 4 --evaluate_per_update -1 --clip_norm 5 \
    --neg_entity_count 100 --max_span_size 8 \
    --graph_neighbors "3 2" --gcn_layers "384 384" \
    --max_batch_nodes 3000 --max_batch_edges 40000 \
    --sampling_processes 4 \
    --edge_weight_threshold 1.0 \
    --gcn_dropout 0.1 \
    --alpha 0.1  \
    --gcn_norm_method "right" \
    --save_cache 0 \
    --load_cache 0 \
    --shuffle 0 \
    --use_char_encoder 1 --use_word_encoder 1 --use_lm_embed 1 \
    --use_size 1 \
    --cased_char 0 --cased_word 0 --cased_lm 1 \
    --concat_span_hid 1  \
    --use_gcn 1 \
    --use_cls 0 \
    --debug 1 \
    --word_embed_dim 100 \
    --cls_embed_dim 768 \
    --lm_dim 768 \
    --char_embed_dim 30 \
    --char_hid_dim 50 \
    --wc_lstm_hid_dim 200 \
    --sent_enc_dropout 0.1 \
    --dropout 0.1 \
    --train_ratio 1.0 \
    --valid_ratio 1.0 \
    --test_ratio 1.0 \
    --do_extra_eval 1 \
    --wv_file $data_dir/ACE04.glove.6B.100d.txt \
    --lm_emb_path $data_dir/ACE04.bert_base_cased_flair.emb.pkl \
    --ent_lm_emb_path $data_dir/ACE04.ent_bert_base_cased_flair.emb.pkl \
    --save_dir $model_dir \
    --save_log_file $result_dir/trnn_"$shell_id".pkl  \
    --save_best_result_file $result_dir/best_results.txt \
    --save_pred_dir $result_dir/"$shell_id"

# --optimizer adam --lr $lr --other_lr $other_lr --weight_decay 1e-8 \
