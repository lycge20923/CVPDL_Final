# CVPDL_Final

### 環境
* 安裝ffpmeg，ubuntu為例：sudo apt install ffmpeg
* python=3.10
* 套件
  ```
    pip install -r requirement.txt
    pip install https://github.com/vBaiCai/python-pesq/archive/master.zip 
    pip install moviepy==1.0.3
    pip install -U typing-extensions
    pip install numba==0.58.1
    pip install librosa==0.10.1
    pip install huggingface-hub==0.19.4
    pip install -U typing-extensions diffusers transformers accelerate xformers
    pip install -U controlnet_aux==0.0.7
  ```
* 執行：
  ```
  python3 main.py --doodle_path {your doodle path} --style '{your style}' --prompt '{your prompt}'
  ```