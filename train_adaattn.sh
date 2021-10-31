python train.py \
--content_path [path to content training dataset] \
--style_path [path to style training dataset] \
--name AdaAttN_test \
--model adaattn \
--dataset_mode unaligned \
--no_dropout \
--load_size 512 \
--crop_size 256 \
--image_encoder_path /other/vgg_normalised.pth \
--gpu_ids 0 \
--batch_size 8 \
--n_epochs 2 \
--n_epochs_decay 3 \
--display_freq 1 \
--display_port 8097 \
--display_env AdaAttN \
--lambda_local 3 \
--lambda_global 10 \
--lambda_content 0 \
--shallow_layer \
--skip_connection_3