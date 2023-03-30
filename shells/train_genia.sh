#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

data_name=genia
archi_name=span_bert_gcn

home_dir=/NAS2020/Workspaces/NLPGroup/juncw
code_dir=$home_dir/codebase/NER/nested_NER/nested_ner_v41
data_dir=$home_dir/database/NER/nested_ner/$data_name

model_dir=$home_dir/database/NER/nested_ner/__models/$data_name/$archi_name
result_dir=$home_dir/database/NER/nested_ner/__results/$data_name/$archi_name

mkdir -p $model_dir
mkdir -p $result_dir

shell_id=41_18

ulimit -n 40000
export LOGLEVEL=INFO

for dropout in 0.2; do
  export LOGFILENAME=$result_dir/"$shell_id"_"$dropout".log
  python -W ignore $code_dir/train.py \
      --data_dir $data_dir --archi "$archi_name"_"$data_name"  \
      --seed 1 --max_sentences 8 --eval_max_sentences 1 \
      --lr_scheduler linear_with_warmup --warmup_steps 0.1 --max_epoch 30 \
      --optimizer adamw --lr 1e-5 --other_lr 3e-4 --weight_decay 1e-8 \
      --update_freq 1 --evaluate_per_update -1 --clip_norm 10 \
      --neg_entity_count 100 --max_span_size 10 \
      --graph_neighbors "5 2" --gcn_layers "384 384" \
      --max_batch_nodes 3000 --max_batch_edges 40000 \
      --sampling_processes 4 \
      --edge_weight_threshold 0.8 \
      --gcn_dropout 0.2 \
      --alpha 0.1 \
      --gcn_norm_method "right" \
      --graph_emb_method "mean" \
      --save_cache 0 \
      --load_cache 0 \
      --shuffle 1 \
      --use_char_encoder 1 --use_word_encoder 1 --use_lm_embed 1 \
      --use_size 1 \
      --cased_char 1 --cased_word 1 --cased_lm 1 \
      --concat_span_hid 1 \
      --use_gcn 1 \
      --use_cls 0 \
      --debug 0 \
      --word_embed_dim 200 \
      --cls_embed_dim 768 \
      --lm_dim 768 \
      --char_embed_dim 30 \
      --char_hid_dim 50 \
      --wc_lstm_hid_dim 300 \
      --sent_enc_dropout $dropout \
      --dropout $dropout \
      --train_ratio 1.0 \
      --valid_ratio 1.0 \
      --test_ratio 1.0 \
      --do_extra_eval 1 \
      --wv_file $data_dir/GENIA.pubmed.200d.txt \
      --lm_emb_path $data_dir/GENIA.biobert_base_cased_flair.emb.pkl \
      --ent_lm_emb_path $data_dir/GENIA.ent_biobert_base_cased_flair.emb.pkl \
      --save_dir $model_dir \
      --save_log_file $result_dir/trnn_"$shell_id".pkl  \
      --save_best_result_file $result_dir/best_results.txt \
      --save_pred_dir $result_dir/"$shell_id"
done
    #--optimizer adam --lr $lr --other_lr $other_lr --weight_decay 1e-8 \
