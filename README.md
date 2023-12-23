# CVPDL_Final

### 環境(遠端執行model)
* 安裝ffpmeg，ubuntu為例：sudo apt install ffmpeg
* 安裝 `lib`相關的套件
  ```
  sudo apt-get install -y \
  libavformat-dev libavcodec-dev libavdevice-dev \
  libavutil-dev libswscale-dev libswresample-dev libavfilter-dev libportaudio2
  ```
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
    pip install -U diffusers transformers accelerate xformers
    pip install -U controlnet_aux==0.0.7
  ```

* 下載pretrain weight
  ```
  mkdir im2wav/pre_trained
  pip install gdown
  gdown 1lCrGsMXqmeKBk-3B3J2jzxNur9olWseb -O im2wav/pre_trained/
  gdown 1v9dmCwrEwkwJhbe2YF3ScM2gjVplSLzt -O im2wav/pre_trained/
  gdown 1UyNBjoxgqBYqA_aYhOu6BHYlkT4CD_M_ -O im2wav/pre_trained/
  ```

* 執行：
  ```
  python3 main.py --doodle_path {your doodle path} --style '{your style}' --prompt '{your prompt}'
  ```

### 環境(本機執行GUI介面)
* python=3.11
* 套件

  ```
    pip install paramiko tk pathlib python-dotenv Pillow
  ```
* 執行：

  ```
  python gui.py
  ```