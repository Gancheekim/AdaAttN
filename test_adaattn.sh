python test.py \
--content_path datasets/contents \
--style_path datasets/styles \
--name AdaAttN \
--model adaattn \
--dataset_mode unaligned \
--load_size 512 \
--crop_size 512 \
--image_encoder_path checkpoints/vgg_normalised.pth \
--gpu_ids 0 \
--skip_connection_3 \
--shallow_layer



# python test.py --name ./AdaAttN_model/AdaAttN --model adaattn --dataset_mode unaligned --image_encoder_path checkpoints/AdaAttN_model/vgg_normalised.pth --gpu_ids 0 --skip_connection_3 --shallow_layer --content_path ./../../content_img/ --style_path ./../../style_img/
