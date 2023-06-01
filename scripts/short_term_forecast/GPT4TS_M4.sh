model_name=GPT4TS

python -u run.py \
  --task_name short_term_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly \
  --model $model_name \
  --data m4 \
  --features M \
  --enc_in 1 \
  --dec_in 1 \
  --c_out 1 \
  --gpt_layer 6 \
  --d_ff 128 \
  --d_model 128 \
  --patch_size 1 \
  --stride 1 \
  --batch_size 16 \
  --des 'Exp' \
  --itr 1 \
  --learning_rate 0.002 \
  --loss 'SMAPE'

# for learning_rate in 0.001
# do
# for layer in 6
# do
# for patch_size in 1
# do
# for stride in 1
# do
# for ln in 1
# do
# for d_ff in 32
# do
# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Yearly' \
#   --model_id m4_Yearly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --ln $ln \
#   --gpt_layer $layer \
#   --d_model 768 \
#   --d_ff $d_ff \
#   --patch_size $patch_size \
#   --stride $stride \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate $learning_rate \
#   --loss 'SMAPE' > logs/short_term_forecast/$model_name/m4_Yearly/new_layer$layer'_'patch_size$patch_size'_'stride$stride'_'d_ff$d_ff'_'ln$ln.logs
# done
# done
# done
# done
# done
# done

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Quarterly' \
#   --model_id m4_Quarterly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --gpt_layer 6 \
#   --d_model 768 \
#   --patch_size 1 \
#   --stride 1 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Daily' \
#   --model_id m4_Daily \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --gpt_layer 6 \
#   --d_model 768 \
#   --patch_size 1 \
#   --stride 1 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Weekly' \
#   --model_id m4_Weekly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --gpt_layer 6 \
#   --d_model 768 \
#   --patch_size 1 \
#   --stride 1 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'

# python -u run.py \
#   --task_name short_term_forecast \
#   --is_training 1 \
#   --root_path ./dataset/m4 \
#   --seasonal_patterns 'Hourly' \
#   --model_id m4_Hourly \
#   --model $model_name \
#   --data m4 \
#   --features M \
#   --enc_in 1 \
#   --dec_in 1 \
#   --c_out 1 \
#   --gpt_layer 6 \
#   --d_model 768 \
#   --patch_size 1 \
#   --stride 1 \
#   --batch_size 16 \
#   --des 'Exp' \
#   --itr 1 \
#   --learning_rate 0.001 \
#   --loss 'SMAPE'
