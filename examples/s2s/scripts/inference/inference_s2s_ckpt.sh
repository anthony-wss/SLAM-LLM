export CKPT=/work/u3937558/SLAM-LLM/exp/s2s_train_v4-Qwen2-0.5b-gpu16-btz6-lr1e-4-fp16-epochs10-whisper_small-latency0-group3/s2s_epoch_4_step_3675

srun --partition dev --gpus-per-node 1 --account MST115022 -N 1 -n 1 \
  singularity exec --nv -B /work/u3937558:/work/u3937558 \
  -B /work/u3937558/SLAM-LLM/src/slam_llm:/opt/SLAM-LLM/src/slam_llm \
  /work/u3937558/SLAM-LLM/slam-omni.sif \
  bash -c "\
    PYTHONPATH=/work/u3937558/SLAM-LLM/src:\$PYTHONPATH \
    echo -e '/work/u3937558/SLAM-LLM/examples/s2s/audio_prompt/en/prompt_3.wav\nq' | python examples/s2s/inference_s2s.py \
      --config-path conf --config-name prompt.yaml \
      hydra.run.dir=$CKPT \
      ++model_config.llm_name=Qwen2-0.5b ++model_config.llm_path=/work/u3937558/models/Qwen2-0.5B ++model_config.llm_dim=896 \
      ++model_config.encoder_name=whisper ++model_config.encoder_projector_ds_rate=5 ++model_config.encoder_path=/work/u3937558/models/whisper/small.pt \
      ++model_config.encoder_dim=768 ++model_config.encoder_projector=linear \
      ++model_config.codec_decoder_path=/work/u3937558/models/CosyVoice-300M-SFT ++model_config.codec_decode=true ++model_config.codec_decoder_type=CosyVoice \
      ++model_config.vocab_config.code_layer=3 ++model_config.vocab_config.total_audio_vocabsize=4160 ++model_config.vocab_config.total_vocabsize=156160 \
      ++model_config.code_type=CosyVoice ++model_config.group_decode=true ++model_config.group_decode_adapter_type=linear \
      ++dataset_config.dataset=speech_dataset_s2s ++dataset_config.input_type=mel ++dataset_config.mel_size=80 ++dataset_config.inference_mode=true \
      ++dataset_config.task_type=s2s \
      ++dataset_config.vocab_config.code_layer=3 ++dataset_config.vocab_config.total_audio_vocabsize=4160 ++dataset_config.vocab_config.total_vocabsize=156160 \
      ++dataset_config.code_type=CosyVoice ++dataset_config.num_latency_tokens=0 ++dataset_config.do_layershift=false \
      ++train_config.model_name=s2s ++train_config.freeze_encoder=true ++train_config.freeze_llm=true ++train_config.batching_strategy=custom ++train_config.num_epochs=1 \
      ++train_config.val_batch_size=1 ++train_config.num_workers_dataloader=0 ++train_config.task_type=s2s \
      ++decode_config.input_text=false ++decode_config.decode_text_only=false ++decode_config.max_new_tokens=3000 ++decode_config.do_sample=false ++decode_config.top_k=0 \
      ++decode_config.top_p=1.0 ++decode_config.temperature=1.0 \
      ++decode_config.text_repetition_penalty=1.2 ++decode_config.audio_repetition_penalty=1.2 ++decode_config.task_type=s2s ++decode_config.do_layershift=false \
      ++decode_config.num_latency_tokens=0 \
      ++log_config.log_file=$CKPT/inference.log ++log_config.online_output_dir=$CKPT/output \
      ++ckpt_path=$CKPT/model.pt ++output_text_only=false ++inference_online=true ++speech_sample_rate=22050 \
      ++audio_prompt_path=/work/u3937558/SLAM-LLM/examples/s2s/audio_prompt/en/prompt_6.wav"