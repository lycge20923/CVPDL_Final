# CVPDL_Final

## Video -> 聲音

### 參考Package
* 來源：im2wav
* 網址：https://github.com/RoySheffer/im2wav.git

### 環境建置
* 建立虛擬環境，使用python=3.10
* 執行以下程式，載```lib```相關的套件(若已載，可忽略)

    ```
    sudo apt-get install -y \
    libavformat-dev libavcodec-dev libavdevice-dev \
    libavutil-dev libswscale-dev libswresample-dev libavfilter-dev libportaudio2
    ```

### 使用(input:video(.mp4)；ouptut:video with audiio(.mp4))
1. Clone 此 Project
    
    ```
    git clone https://github.com/lycge20923/CVPDL_Final.git
    cd im2wav
    ```

2. 下載及升級相關套件(請使用這裡上傳的，因為原版im2wav給的```requirements.txt```裡面有蠻多packages會衝突)
    ```
    cd im2wav
    pip install -r requirements.txt
    pip install https://github.com/vBaiCai/python-pesq/archive/master.zip 
    pip install moviepy==1.0.3
    pip install typing-extensions --upgrade
    ```

3. 下載pre-trained Weights(若已載請忽略)

    ```
    mkdir pre_trained
    pip install gdown
    gdown 1lCrGsMXqmeKBk-3B3J2jzxNur9olWseb -O pre_trained/
    gdown 1v9dmCwrEwkwJhbe2YF3ScM2gjVplSLzt -O pre_trained/
    gdown 1UyNBjoxgqBYqA_aYhOu6BHYlkT4CD_M_ -O pre_trained/
    ```

4. 在python檔中import ```VideoToAudio.py```

    ```
    cd .. # 回到上一層資料夾

    #vim
    import VideoToAudio.py as VT
    VT.run('sample.mp4') # replace the ```sample.mp4``` to your file
    ```

### 補充：原版project ```im2wav``` 的使用

1. 蒐集某Video的```pickle```檔
    1. 建立並進入```run```資料夾後(後續都在這裡執行)
    2. 將蒐集到的Video存在一個資料夾。此project以預先存好示範影片，位於```../Data/examples/video```中
    3. 執行```collect_video_CLIP.py```，蒐集某資料夾下所有Video對應的CLIP representations，即```pickle```檔，相關參數：
        1. ```-save_dir```：指定存取地。否則，會直接在```run```資料夾中建立```video_CLIP```資料夾
        2. ```-videos_dir```：video來源，下方例子是直接取im2wav此project中即存放在```../Data/examples/video```的
    
    ```
    mkdir run && cd run
    python ../Data/preprocess/collect_video_CLIP.py \
    -videos_dir ../Data/examples/video
    ```

2. 執行Video Condition Sampling，相關參數：

    1. ```-bs```：不確定是啥，好像是Batch相關的係數
    2. ```-save_dir```：後續儲存音檔的主目錄，default值為"samples"
    3. ```-experiment_name```：在儲存音檔主目錄下一層資料夾，沒有Default值，一定要令
    4. ```-CLIP_dir```：輸入的pickle檔來源資料夾，沒有defalut值。
    5. ```-models```：若要train by myself，可以自己令自己的Model。但我們使用pretrained model，令im2wav就好
    
    ```
    python ../models/sample.py \
    -bs 2 \
    -save_dir samples \
    -experiment_name video_CLIP \
    -CLIP_dir video_CLIP \
    -models im2wav
    ```

3. 相關音檔即可在 ```samples/video_CLIP/k_top0_p_top0/im2wav```找到囉

