pip install -U huggingface_hub hf_transfer -i https://pypi.tuna.tsinghua.edu.cn/simple

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --repo-type dataset --resume-download ZhiYiTian/GTA5_dataset --local-dir /data/tangpeiyuan/