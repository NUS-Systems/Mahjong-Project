from huggingface_hub import snapshot_download

# 设置你的 API key
api_key = "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"

model_repos = {
    "meta-llama/Llama-3.2-1B" : "962c3106fec9c7b4d6cc6faa5de51a1b9aee137c",
    "meta-llama/Llama-3.2-3B" : "251c4ef1ac2fe50aa6a0948d742287aaeb9bb159",
    "meta-llama/Llama-3.1-8B" : "3d5d993573e28422665efa60478b6d1d41b13ace",
    "Qwen/Qwen2.5-7B-Instruct" : "d1e200fcf95ef0d4326873ddf63e5562d5f1fdbb",
    "mistralai/Ministral-8B-Instruct-2410" : "05894f4c1a269ccce053c854da977fe06cad2d17",
}

for model_repo in model_repos.keys():
    blob_id = model_repos[model_repo]


    # 设置下载文件的目标文件夹
    target_folder = f"/mydata/Models/{model_repo.split('/')[-1]}"

    # 下载整个模型仓库到指定文件夹
    snapshot_download(
        repo_id=model_repo,
        revision=blob_id,
        local_dir=target_folder,
        use_auth_token=api_key,
        local_dir_use_symlinks=False  # 确保文件直接下载到目标文件夹，而不是通过符号链接
    )

    print(f"Model downloaded to {target_folder}")
