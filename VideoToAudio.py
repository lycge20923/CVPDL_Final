'''
This is the python file for transfer video into audio and combine them
To use this python file, try
```
import VideoToAudio.py as VT
VT.run('generated.mp4')
```
'''

# Import the package path
# os.sep:"/"" ; os.pardir:""..""
# sys.path.append: add the module to the import path  
import os, sys
modules_dir = os.path.normpath(os.getcwd()+os.sep+'im2wav')
sys.path.append(modules_dir)

# Import the default required packages
import skvideo.io
import torch
from PIL import Image
import clip
import numpy as np
import pickle
import glob
import traceback as tb
import math
import time
from moviepy.editor import *
import shutil

from im2wav_utils import *
from models.hparams import CLIP_VERSION
from models.utils.dist_utils import setup_dist_from_mpi
from models.utils.torch_utils import empty_cache
from models.trained_models import *
from models.hparams import SAMPLE_RATE, VIDEO_TOTAL_LENGTHS, VIDEO_FPS

'''
From the `Collect_video_CLIP.py`
'''
def convert(s):
    try:
        return float(s)
    except ValueError:
        num, denom = s.split('/')
        return float(num) / float(denom)

def read_video(video, global_info):
    try:
        video_name = os.path.basename(video).split('.')[0]
        videodata = skvideo.io.vread(video)
        videometadata = skvideo.io.ffprobe(video)
        frame_rate = videometadata['video']['@avg_frame_rate']
        frame_num = videodata.shape[0]
        frames_in_sec = convert(frame_rate)
        length_in_secs = frame_num / frames_in_sec

        if global_info["videos_length"] is not None:
            if length_in_secs != global_info["videos_length"]:
                print(f"{video} video length: {frame_num} frames {length_in_secs} secs filtered\n\n")
                # os.remove(video)
                return [None, None, None, video_name]
        return [videodata, length_in_secs, frame_num, video_name]

    except Exception as e:
        err_msg = '{} Error while reading video: {}; \n{} {}'.format(video, e, tb.format_exc(),"\n-----------------------------------------------------------------------\n")
        print(err_msg)
        # os.remove(video)
        return [None, None, None, None]


def get_video_clip(video, device, model, global_info):
    with torch.no_grad():
        images = torch.cat([global_info["preprocess"](frame).unsqueeze(0).to(device) for frame in video])
        image_features = model.encode_image(images)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        image_features = image_features.cpu().numpy()
        return image_features

def handle_video(video, global_info):
    try:
        video_name = os.path.basename(video).split('.')[0]
        save_dir = global_info["save_CLIP_dir"]
        pickle_name = f"{save_dir}/{video_name}.pickle"
        if os.path.exists(pickle_name):
            return
        videodata, length_in_secs, frame_num, video_name = read_video(video, global_info)
        video_to_embed = [Image.fromarray(frame) for frame in videodata]
        image_features = get_video_clip(video_to_embed, global_info["device"], global_info["model"], global_info)
        file2CLIP = {}
        file2CLIP[video_name] = image_features
        with open(pickle_name, 'wb') as handle:
            pickle.dump(file2CLIP, handle, protocol=pickle.HIGHEST_PROTOCOL)

    except Exception as e:
        err_msg = 'Error while processing video {}: {}; {}'.format(video, e, tb.format_exc())
        print(err_msg)

'''
From the `Sample.py`
'''
# Break total_length into hops/windows of size n_ctx separated by hop_length
def get_starts(total_length, n_ctx, hop_length):
    starts = []
    for start in range(0, total_length - n_ctx + hop_length, hop_length):
        if start + n_ctx >= total_length:
            # Last hop could be smaller, we make it n_ctx to maximise context
            start = total_length - n_ctx
        starts.append(start)
    return starts


def adjust_y(y, prior, start, total_sample_length):
    if y is None:
        print(f"y is None")
        return y
    if len(y.shape) == 3: # (batch, frames, y ) y[2] = [total_length, offset, sample_length, clip]
        y[:, :, 2] = int(prior.sample_length)  # Set sample_length to match this level
        offset = int(start * prior.raw_to_tokens)
        y[:, :, 1:2] = offset
        offset_in_sec, length_in_sec = offset / float(SAMPLE_RATE), prior.sample_length / float(SAMPLE_RATE)
        print(offset_in_sec ,VIDEO_FPS)
        start = int(offset_in_sec * VIDEO_FPS)
        end = int(start + length_in_sec * VIDEO_FPS)
        print(f"using frames [{start}, {end}] out of the total {y.shape[1]} frames {(end - start)/y.shape[1]} =? {prior.sample_length / total_sample_length}")
        y = y[:, start:end + 1]
    elif len(y.shape) == 2: # (batch, y) y[1] = [total_length, offset, sample_length, clip]
        y[:, 2] = int(prior.sample_length)  # Set sample_length to match this level
        offset = int(start * prior.raw_to_tokens)  # Set offset
        y[:, 1:2] = offset
    return y


def multi_level_sample_window(priors, y, y_video, n_samples, top_k, top_p, cfg_s, hop_fraction, total_sample_length, sliding_window=False):
    sample_levels = list(range(len(priors)))
    zs = [torch.zeros(n_samples, 0, dtype=torch.long, device=device) for _ in range(len(priors))]
    xs = []
    for level in reversed(sample_levels):
        prior = priors[level]
        if prior is None:
            continue
        if torch.cuda.is_available():
            prior.cuda()

        if prior.video_clip_emb:
            y_hat = y_video
        else:
            y_hat = y

        empty_cache()

        assert total_sample_length % prior.raw_to_tokens == 0, f"Expected sample_length {total_sample_length} to be multiple of {prior.raw_to_tokens}"

        if sliding_window:
            # Set correct total_length, hop_length, labels and sampling_kwargs for level
            total_length = total_sample_length//prior.raw_to_tokens
            hop_length = int(hop_fraction[level]*prior.n_ctx)

            for start in get_starts(total_length, prior.n_ctx, hop_length):
                end = start + prior.n_ctx
                z = zs[level][:, start:end]

                sample_tokens = (end - start)
                conditioning_tokens, new_tokens = z.shape[1], sample_tokens - z.shape[1]
                print(f"Sampling {sample_tokens} tokens for [{start},{start + sample_tokens}]. Conditioning on {conditioning_tokens} tokens z.shape={z.shape}")
                if new_tokens <= 0:
                    # Nothing new to sample
                    continue
                # get z_conds from level above
                z_conds = prior.get_z_conds(zs, start, end)

                # set y offset, sample_length and lyrics tokens
                y_cur = adjust_y(y_hat.clone(), prior, start, global_info['total_sample_length'])
                empty_cache()

                z = prior.sample(n_samples=n_samples, z=z, z_conds=z_conds, y=y_cur, top_k=top_k, top_p=top_p, cfg_s=cfg_s)
                # Update z with new sample
                z_new = z[:, -new_tokens:]
                zs[level] = torch.cat([zs[level], z_new], dim=1)
        else:
            start = 0
            end = start + prior.n_ctx
            z = zs[level][:, start:end]
            # get z_conds from level above
            z_conds = prior.get_z_conds(zs, start, end)
            y_cur = adjust_y(y_hat, prior, start, global_info['total_sample_length'])
            zs[level] = prior.sample(n_samples=n_samples, z=z, z_conds=z_conds, y=y_cur, top_k=top_k, top_p=top_p, cfg_s=cfg_s)

        prior.cpu()
        empty_cache()

        x = prior.decode(zs[level:], start_level=level, bs_chunks=zs[level].shape[0])  # Decode sample
        xs.append(x)
    return xs


def get_y(video, pos ,clip_emb):
    if video:
        pose_tiled = np.tile(pos, (clip_emb.shape[0], 1))
        y = np.concatenate([pose_tiled, clip_emb], axis=1, dtype=np.float32)
    else:
        y = np.concatenate([pos, clip_emb.flatten()], dtype=np.float32)
    return y


def save_samples(global_info, sliding_window=False):
    #resultsDir = os.path.join(global_info['save_dir'], global_info["experiment_name"], global_info["resultsDir"], global_info["model_name"])
    resultsDir = os.path.join(global_info['save_dir'], global_info["resultsDir"], global_info["model_name"])
    first = 0
    if not os.path.exists(resultsDir):
        os.makedirs(resultsDir)
        for level in [0,1]:
            os.makedirs(f"{resultsDir}/l{level}")

    if global_info["first"] is not None:
        if first < global_info["first"]:
            first = global_info["first"]

    with torch.no_grad():
        print(f"using {device} global_info:{global_info}")

        TRIES = 10
        for tr in range(TRIES):
            try:
                global_info["vq_cp"] = 'im2wav/pre_trained/vqvae.tar'
                global_info["prior_cp"] = 'im2wav/pre_trained/low.tar'
                global_info["up_cp"] = 'im2wav/pre_trained/up.tar'
                vqvae = get_model_from_checkpoint(global_info["vq_cp"], device)
                prior = get_model_from_checkpoint_prior(global_info["prior_cp"], vqvae, device)
                upsampler = get_model_from_checkpoint_prior(global_info["up_cp"], vqvae, device)
                break
            except Exception as e:
                print(tr, ": ------------------ \n", e)
                time.sleep(5)

        priors = [upsampler, prior]

        print(global_info["model_name"], f"starting from {first} top_k:", global_info["top_k"], "top_p:", global_info["top_p"] ,"  --------------------------------------------------------------------")
        for i in range(global_info["num_batches"]):
            if i * global_info["bs"] < first:
                continue

            start, end = i * global_info["bs"], (i + 1) * global_info["bs"]
            end = min(end, ys.shape[0])
            required = 0
            for m in range(start, end):
                if not os.path.exists(f"{resultsDir}/l0/{file_names[m]}.wav") or not os.path.exists(f"{resultsDir}/l1/{file_names[m]}.wav"):
                    required += 1
            if required == 0:
                print(f"skipping [{start}, {end}]")
                continue
            else:
                print(f"required {required} in [{start}, {end}]")

            y = ys[start:end]
            y_video = ys_video[start:end]

            y = y.to(device, non_blocking=True)
            y_video = y_video.to(device, non_blocking=True)

            xs = multi_level_sample_window(priors, y, y_video, y.shape[0], global_info["top_k"], cfg_s=global_info["cfg_s"], top_p=global_info["top_p"], hop_fraction=global_info["hop_fraction"], total_sample_length=global_info['total_sample_length'], sliding_window=sliding_window)

            for level, x_sample in enumerate(xs):
                for j in range(y.shape[0]):
                    index = i * global_info["bs"] + j
                    if index < first:
                        continue
                    audio = x_sample[j, :, 0].cpu().numpy()
                    name = f"{resultsDir}/l{level}/{file_names[index]}.wav"
                    soundfile.write(name,audio, samplerate=SAMPLE_RATE, format='wav')
    return f"{resultsDir}/l1/{file_names[m]}.wav"

def Combine(video_path, audio_path):
    audio = AudioFileClip(audio_path)
    video = VideoFileClip(video_path)
    video = video.set_audio(audio)
    video.write_videofile(video_path)
    after_pth = video_path.split('/')[-1][:-4]+'_with_audio.mp4'
    os.rename(video_path, after_pth)
    try:
        shutil.rmtree('audio')
        shutil.rmtree('video_CLIP')
        shutil.rmtree('videos')
    except:
        pass

def run(video_path):
    global global_info
    global_info = {
    'save_CLIP_dir':'video_CLIP',
    'video':video_path,
    'videos_dir': 'videos',
    'videos_length': None,
    'bs': 10,
    'multi_thread': None,
    'experiment_name': 'no_name',
    'save_dir': 'audio',
    'model_name': 'my_model',
    'vq_cp': 'im2wav/pre_trained/vqvee.tar',
    'prior_cp': 'im2wav/pre_trained/low.tar',
    'up_cp': 'im2wav/pre_trained/up.tar',
    'resultsDir': '',
    'first': None,
    'cfg_s': 3.0,
    'wav_per_object': 120,
    'p_grid': [0],
    'k_grid': [0],
    'CLIP_dir': 'video_CLIP',
    'CLIP_dict': None,
    'models': ['im2wav'],
    'hop_fraction': None,
    'sample_length': 65536,
    'sliding_window_total_sample_length': None,
    'device':torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}
    '''
    #From the `Collect_video_CLIP.py`
    '''
    print("running CLIP collection")

    if global_info["multi_thread"] is not None:
        import multiprocessing as mp
        import atexit
        atexit.register(lambda: os.system('stty sane') if sys.stdin.isatty() else None)

    global_info["device"] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(global_info["save_CLIP_dir"], exist_ok=True)
    os.makedirs(global_info["videos_dir"], exist_ok=True)
    ori_path = global_info['video']
    video_path = os.path.join(global_info["videos_dir"],global_info['video'])
    #os.rename(global_info['video'], video_path)
    os.system(f'cp "{ori_path}" "{video_path}"')

    global_info["videos"] = glob.glob(os.path.join(global_info["videos_dir"], '*.mp4'))
    global_info["videos_num"] = len(global_info["videos"])

    global_info["num_batches"] = math.ceil(global_info["videos_num"] / float(global_info["bs"]))
    print("assert ", (global_info["num_batches"]-1) * global_info["bs"], "<", global_info["videos_num"], "<", global_info["num_batches"] * global_info["bs"])
    print("using " , global_info["videos_num"], " videos")

    sys.stdout.flush()
    with torch.no_grad():
        global_info["model"], global_info["preprocess"] = clip.load(CLIP_VERSION, device=global_info["device"])
        if global_info["multi_thread"] is not None:
            max_num_workers = mp.cpu_count()
            num_workers = min(max_num_workers,10)
            print(f"using {num_workers} workers max_num_workers={max_num_workers}")
            for b in range(global_info["num_batches"]):
                num_batches = global_info["num_batches"]
                print(f"batch {b+1} out of {num_batches} ")
                batch_videos = global_info["videos"][b*global_info["bs"]:(b+1)*global_info["bs"]]
                with mp.Pool(num_workers) as pool:
                    try:
                        for i, video in enumerate(batch_videos):
                            pool.apply_async(handle_video, args=(video, global_info))
                    except Exception as e:
                        err_msg = 'Encountered error in {} at line: {}'
                        sys.exit(err_msg.format(video, e))
                    finally:
                        pool.close()
                        pool.join()
                sys.stdout.flush()
        else:
            for i, video in enumerate(global_info["videos"]):
                handle_video(video, global_info)
                if i % 1000 == 0:
                    print(f"finished {i+1}")
                    sys.stdout.flush()
    print("finished Generating the Clip file")

    '''
    #From the `Sample.py`
    '''
    global device 
    rank, local_rank, device = setup_dist_from_mpi(port=(29500 + np.random.randint(99)))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = [name2model[model_name] for model_name in global_info["models"]]
    if len(models):
        global_info['sample_length'] = models[0]["sample_length"]

    if global_info['sliding_window_total_sample_length'] is None:  #single generation not sliding window
        sliding_window = False
        global_info['total_sample_length'] = global_info['sample_length']
    else:
        sliding_window = True
        global_info['total_sample_length'] = global_info['sliding_window_total_sample_length']


    duration_in_sec = global_info['total_sample_length']/SAMPLE_RATE
    offset = 0.0
    pos = np.array([VIDEO_TOTAL_LENGTHS, offset, global_info['sample_length']], dtype=np.int64) # all current generation models should share the same sample_length

    if global_info["CLIP_dir"]:
        global ys, ys_video, video_names
        ys, ys_video, video_names = [], [], []
        CLIP_paths = glob.glob(os.path.join(global_info["CLIP_dir"], '*.pickle'))
        CLIP_paths = sorted(CLIP_paths) # to be able to rerun killed jobs based on indexes
        for CLIP_path in CLIP_paths:
            video_name = os.path.basename(CLIP_path).split('.')[0]
            if not os.path.exists(CLIP_path):
                continue
            with open(CLIP_path, 'rb') as handle:
                clip_emb = pickle.load(handle)[video_name]
            clip_emb= clip_emb[:int(duration_in_sec*VIDEO_FPS)]
            mean_sample_clip_emb = np.mean(clip_emb, axis=0)
            pose_tiled = np.tile(pos, (clip_emb.shape[0], 1))
            y_video = np.concatenate([pose_tiled, clip_emb], axis=1, dtype=np.float32)
            y = np.concatenate([pos, mean_sample_clip_emb.flatten()], dtype=np.float32)
            ys.append(y)
            ys_video.append(y_video)
            video_names.append(video_name)
        ys = torch.from_numpy(np.stack(ys))
        ys_video = torch.from_numpy(np.stack(ys_video))
        global file_names
        file_names = video_names
    else:
        with open(global_info["CLIP_dict"], 'rb') as handle:
            CLIP_dict = pickle.load(handle)
            CLIP = CLIP_dict["image"]
        objects = list(CLIP.keys())
        chosen_objects = {"objects": [], "indexs": [], "clip_emb": [], "total_length": VIDEO_TOTAL_LENGTHS}
        clip_emb_all = []
        class_list = CLIP.keys()
        class_list = list(class_list)
        class_list.sort()
        for class_name in class_list:
            chosen_objects["objects"] += ([class_name] * global_info["wav_per_object"])
            class_images_num = CLIP[class_name].shape[0]
            class_indices = list(range(class_images_num)) * int(np.ceil(global_info["wav_per_object"] / class_images_num))
            class_indices = class_indices[: global_info["wav_per_object"]]
            class_indices.sort()
            class_clip_emb = [CLIP[class_name][class_indices[i]] for i in range(len(class_indices))]
            chosen_objects["indexs"] += class_indices
            clip_emb_all += class_clip_emb
        clip_emb_all = np.array(clip_emb_all)
        ys = [get_y(video=False, pos=pos, clip_emb=clip_emb_all[i]) for i in range(clip_emb_all.shape[0])]
        ys = torch.from_numpy(np.stack(ys))
        ys_video = ys.reshape((ys.shape[0], 1, ys.shape[1]))
        file_names = []
        for i in range(ys.shape[0]):
            class_cur = chosen_objects["objects"][i]
            index_cur = chosen_objects["indexs"][i]
            name = f"{i}_{class_cur}_{index_cur}"
            file_names.append(name)
        global_info["num_batches"] = math.ceil(len(chosen_objects["objects"]) / float(global_info["bs"]))
    global_info["num_batches"] = math.ceil(ys.shape[0] / float(global_info["bs"]))
    if len(models) !=0:
        for k in global_info["k_grid"]:
            for p in global_info["p_grid"]:
                global_info["top_k"] = k
                global_info["top_p"] = p
                global_info["resultsDir"] = ""
                if len(global_info["k_grid"]) + len( global_info["p_grid"]):
                    global_info["resultsDir"]+=f"k_top{k}_p_top{p}"
                for model in models:
                    hps_cur = global_info.copy()
                    for key in model:
                        hps_cur[key] = model[key]
                    audio_pth = save_samples(hps_cur, sliding_window=sliding_window)
    else:
        audio_pth = save_samples(global_info, sliding_window=sliding_window)
    print("finished")

    Combine(video_path, audio_pth)