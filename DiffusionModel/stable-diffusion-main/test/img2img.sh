cd ~/Repos/StableDiffusionOnlineStorage/FTSD/stable-diffusion-main/

python scripts/img2img.py \
    --prompt "" \
    --init-img "~/Data/PACSm/art_painting/elephant/pic_001.jpg" \
    --outdir "~/Repos/StableDiffusionOnlineStorage/FTSD/stable-diffusion-main/test/img2img"\
    --n_samples 8\
    --scale 1\
    --strength 1.0\
    --config "~/Repos/StableDiffusionOnlineStorage/FTSD/stable-diffusion-main/logs/2022-10-22T18-04-49_ours/configs/2022-10-22T18-04-49-project.yaml"\
    --ckpt "~/Repos/StableDiffusionOnlineStorage/FTSD/stable-diffusion-main/logs/2022-10-22T18-04-49_ours/checkpoints/last.ckpt"