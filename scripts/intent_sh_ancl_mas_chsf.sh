#!/bin/bash

# set -Eeuo pipefail

# Initialise the following: res_path, lr_array, decay, acc_drop_threshold, growth, start_alpha_lamb
note=$1 #random10
randid=$2 #10
seed=$3 #0
custom_max_lamb=$4
dataset='hwu64'
lr_array=(0.00003 0.0003)
decay=0.9
acc_drop_threshold=$5
growth=0.1
start_alpha_lamb=0.01
res_path=""

id=0
printf "\n\nRunning search for task 0\n\n"
lr_id=0
for lr in "${lr_array[@]}"
do
	((lr_id++))
	printf "\n\nLR Iteration $lr\n\n"
	mkdir -p  ${res_path}${id}_gold.${lr_id}/
	python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --learning_rate $lr --remove_lr_schedule True --remove_wd True --ancl True --break_after_task 0 --my_save_path ${res_path}${id}_gold.${lr_id}/
done

python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
best_lr_id=$?
past_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
past_lamb=0
past_alpha_lamb=0

start_model_path="${res_path}${id}_gold.${best_lr_id}/"

id_array=(1 2 3 4 5)
for id in "${id_array[@]}"
do
	printf "\n\nRunning search for task $id\n\n"
	lr_id=0
	for lr in "${lr_array[@]}"
	do
		((lr_id++))
		printf "\n\nLR Iteration $lr\n\n"
		custom_lamb="$past_lamb,0"
		custom_alpha_lamb="$past_alpha_lamb,0"
		custom_lr="$past_lr,$lr"
		mkdir -p  ${res_path}${id}_gold.${lr_id}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --remove_lr_schedule True --remove_wd True --custom_lamb $custom_lamb --custom_alpha_lamb $custom_alpha_lamb --ancl True --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}_gold.${lr_id}/ --start_at_task $id --start_model_path $start_model_path
	done
	
	python3 FABR/return_best_lr.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --dataset $dataset --max_lr_id $lr_id --tid $id
	best_lr_id=$?
	best_lr=${lr_array[$best_lr_id-1]}  # -1 for array indexing
	past_lr="$past_lr,$best_lr"
	python3 FABR/calc_max_lamb.py --my_save_path ${res_path}${id}_gold --rand_idx $randid --seed $seed --best_lr_id $best_lr_id --best_lr $best_lr --tid $id --tid $id --custom_max_lamb $custom_max_lamb
	start_lamb=$(<${res_path}${id}_gold_max_lamb.txt)
	if [ "$id" -gt 1 ]; then
		start_lamb=$best_lamb
	fi

	## Lamb
	lamb=$start_lamb
	lamb_i=0
	found_best=false
	while [ $found_best=false ]
	do
		((lamb_i++))
		custom_lr=$past_lr
		custom_lamb="$past_lamb,$lamb"
		custom_alpha_lamb="$past_alpha_lamb,0"
		printf "\n\nLamb Iteration $custom_lamb $custom_alpha_lamb\n\n"
		mkdir -p  ${res_path}${id}.${lamb_i}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --remove_lr_schedule True --remove_wd True --custom_lamb $custom_lamb --custom_alpha_lamb $custom_alpha_lamb --ancl True --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${lamb_i}/ --start_at_task $id --start_model_path $start_model_path
		python3 FABR/calc_next_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --decay $decay --acc_drop_threshold $acc_drop_threshold --tid $id
		found_best=`cat ${res_path}${id}.${lamb_i}_foundbestlamb.txt`
		python3 FABR/plot_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --lamb_i $lamb_i --lamb $lamb --acc_drop_threshold $acc_drop_threshold --tid $id
		if [ $found_best = found ]; then
			best_lamb=$lamb
			best_lamb_i=$lamb_i
			break
		fi
		lamb=`cat ${res_path}${id}_next_lamb.txt`
	done
	
	past_lamb="$past_lamb,$best_lamb"
	
	## Alpha lamb
	alpha_lamb=$start_alpha_lamb
	alpha_lamb_i=0
	found_best=false
	while [ $found_best=false ]
	do
		((alpha_lamb_i++))
		custom_lr=$past_lr
		custom_lamb=$past_lamb
		custom_alpha_lamb="$past_alpha_lamb,$alpha_lamb"
		printf "\n\nAlpha Lamb Iteration $custom_lamb $custom_alpha_lamb\n\n"
		mkdir -p ${res_path}${id}.${best_lamb_i}.${alpha_lamb_i}/
		python  FABR//run.py --bert_model 'bert-base-uncased' --experiment hwu64 --approach bert_adapter_ewc_ancl --imp function --baseline ewc_ancl --backbone bert_adapter --note $note --idrandom $randid --seed $seed --scenario cil --use_rbs True --train_batch_size 128 --num_train_epochs 50 --valid_loss_es 0.002 --lr_patience 5 --custom_lr $custom_lr --remove_lr_schedule True --remove_wd True --custom_lamb $custom_lamb --custom_alpha_lamb $custom_alpha_lamb --ancl True --break_after_task $id --save_alpharel True --my_save_path ${res_path}${id}.${best_lamb_i}.${alpha_lamb_i}/ --start_at_task $id --start_model_path $start_model_path
		python3 FABR/calc_next_alpha_lamb.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lr_id $best_lr_id --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --alpha_lamb $alpha_lamb --growth $growth --tid $id
		found_best=`cat ${res_path}${id}.${best_lamb_i}.${alpha_lamb_i}_foundbestalphalamb.txt`
		python3 FABR/plot_alpha_lamb_results.py --my_save_path ${res_path}${id} --rand_idx $randid --seed $seed --dataset $dataset --best_lamb_i $best_lamb_i --alpha_lamb_i $alpha_lamb_i --alpha_lamb $alpha_lamb --tid $id
		if [ $found_best = found ]; then
			best_alpha_lamb=$alpha_lamb
			best_alpha_lamb_i=$alpha_lamb_i
			break
		fi
		alpha_lamb=`cat ${res_path}${id}_next_alpha_lamb.txt`
	done
	
	past_alpha_lamb="$past_alpha_lamb,$best_alpha_lamb"
	start_model_path="${res_path}${id}.${best_lamb_i}.${best_alpha_lamb_i}/"
done


