import os
import sys
# single thread doubles cuda performance - needs to be set before torch import
if any(arg.startswith('--execution-provider') for arg in sys.argv):
    os.environ['OMP_NUM_THREADS'] = '1'
# reduce tensorflow log level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
import tempfile
from typing import List
import platform
import signal
import shutil
import argparse
import torch
import onnxruntime
import tensorflow
from dotenv import load_dotenv

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.face_analyser import get_one_face
import cv2
from modules.utilities import has_image_extension, is_image, is_video, detect_fps, create_video, extract_frames, get_temp_frame_paths, restore_audio, create_temp, move_temp, clean_temp, normalize_output_path

if 'ROCMExecutionProvider' in modules.globals.execution_providers:
    del torch

warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
warnings.filterwarnings('ignore', category=UserWarning, module='torchvision')


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument('-s', '--source', help='select an source image', dest='source_path')
    program.add_argument('-t', '--target', help='select an target image or video', dest='target_path')
    program.add_argument('-o', '--output', help='select output file or directory', dest='output_path')
    program.add_argument('--frame-processor', help='pipeline of frame processors', dest='frame_processor', default=['face_swapper'], choices=['face_swapper', 'face_enhancer'], nargs='+')
    program.add_argument('--keep-fps', help='keep original fps', dest='keep_fps', action='store_true', default=False)
    program.add_argument('--keep-audio', help='keep original audio', dest='keep_audio', action='store_true', default=True)
    program.add_argument('--keep-frames', help='keep temporary frames', dest='keep_frames', action='store_true', default=False)
    program.add_argument('--many-faces', help='process every face', dest='many_faces', action='store_true', default=False)
    program.add_argument('--nsfw-filter', help='filter the NSFW image or video', dest='nsfw_filter', action='store_true', default=False)
    program.add_argument('--map-faces', help='map source target faces', dest='map_faces', action='store_true', default=False)
    program.add_argument('--mouth-mask', help='mask the mouth region', dest='mouth_mask', action='store_true', default=False)
    program.add_argument('--video-encoder', help='adjust output video encoder', dest='video_encoder', default='libx264', choices=['libx264', 'libx265', 'libvpx-vp9'])
    program.add_argument('--video-quality', help='adjust output video quality', dest='video_quality', type=int, default=18, choices=range(52), metavar='[0-51]')
    program.add_argument('-l', '--lang', help='Ui language', default="en")
    program.add_argument('--live-mirror', help='The live camera display as you see it in the front-facing camera frame', dest='live_mirror', action='store_true', default=False)
    program.add_argument('--live-resizable', help='The live camera frame is resizable', dest='live_resizable', action='store_true', default=False)
    program.add_argument('--max-memory', help='maximum amount of RAM in GB', dest='max_memory', type=int, default=suggest_max_memory())
    program.add_argument('--execution-provider', help='execution provider', dest='execution_provider', default=['cpu'], choices=suggest_execution_providers(), nargs='+')
    program.add_argument('--execution-threads', help='number of execution threads', dest='execution_threads', type=int, default=suggest_execution_threads())
    program.add_argument('--det-thresh', help='face detection threshold (0-1, higher=stricter)', dest='det_thresh', type=float, default=0.5)
    program.add_argument('--assign-sim', help='min cosine similarity to assign faces to clusters (0-1)', dest='assign_sim', type=float, default=0.6)
    program.add_argument('--cluster-method', help='clustering model selection method', dest='cluster_method', choices=['elbow','silhouette'], default='elbow')
    program.add_argument('--cluster-max-k', help='max K to search for clustering', dest='cluster_max_k', type=int, default=10)
    program.add_argument('--cluster-min-size', help='minimum faces per cluster to keep', dest='cluster_min_size', type=int, default=1)
    group_nb = program.add_mutually_exclusive_group()
    group_nb.add_argument('--nano-banana', help='enable Gemini preprocessing on source image', dest='nano_banana', action='store_true')
    group_nb.add_argument('--no-nano-banana', help='disable Gemini preprocessing', dest='nano_banana', action='store_false')
    program.set_defaults(nano_banana=None)
    program.add_argument('--nano-target', help='also preprocess target if it is an image (videos skipped)', dest='nano_target', action='store_true', default=False)
    program.add_argument('--nano-model', help='Gemini image model', dest='nano_model', default=None)
    program.add_argument('--nano-prompt', help='prompt for nano banana preprocessing', dest='nano_prompt', default=None)
    program.add_argument('-v', '--version', action='version', version=f'{modules.metadata.name} {modules.metadata.version}')

    # register deprecated args
    program.add_argument('-f', '--face', help=argparse.SUPPRESS, dest='source_path_deprecated')
    program.add_argument('--cpu-cores', help=argparse.SUPPRESS, dest='cpu_cores_deprecated', type=int)
    program.add_argument('--gpu-vendor', help=argparse.SUPPRESS, dest='gpu_vendor_deprecated')
    program.add_argument('--gpu-threads', help=argparse.SUPPRESS, dest='gpu_threads_deprecated', type=int)

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(modules.globals.source_path, modules.globals.target_path, args.output_path)
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(args.execution_provider)
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang
    modules.globals.det_thresh = max(0.0, min(1.0, args.det_thresh))
    modules.globals.assign_min_similarity = max(0.0, min(1.0, args.assign_sim))
    modules.globals.cluster_method = args.cluster_method
    modules.globals.cluster_max_k = max(2, int(args.cluster_max_k))
    modules.globals.cluster_min_cluster_size = max(1, int(args.cluster_min_size))
    # Nano Banana flags
    modules.globals.nano_banana_user_override = args.nano_banana is not None
    if args.nano_banana is not None:
        modules.globals.enable_nano_banana = args.nano_banana
    modules.globals.nano_banana_on_target = args.nano_target
    if args.nano_model:
        modules.globals.nano_banana_model = args.nano_model
    if args.nano_prompt:
        modules.globals.nano_banana_prompt = args.nano_prompt

    #for ENHANCER tumbler:
    if 'face_enhancer' in args.frame_processor:
        modules.globals.fp_ui['face_enhancer'] = True
    else:
        modules.globals.fp_ui['face_enhancer'] = False

    # translate deprecated args
    if args.source_path_deprecated:
        print('\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m')
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(args.source_path_deprecated, modules.globals.target_path, args.output_path)
    if args.cpu_cores_deprecated:
        print('\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == 'apple':
        print('\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['coreml'])
    if args.gpu_vendor_deprecated == 'nvidia':
        print('\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['cuda'])
    if args.gpu_vendor_deprecated == 'amd':
        print('\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m')
        modules.globals.execution_providers = decode_execution_providers(['rocm'])
    if args.gpu_threads_deprecated:
        print('\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m')
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [execution_provider.replace('ExecutionProvider', '').lower() for execution_provider in execution_providers]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [provider for provider, encoded_execution_provider in zip(onnxruntime.get_available_providers(), encode_execution_providers(onnxruntime.get_available_providers()))
            if any(execution_provider in encoded_execution_provider for execution_provider in execution_providers)]


def suggest_max_memory() -> int:
    if platform.system().lower() == 'darwin':
        return 4
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if 'DmlExecutionProvider' in modules.globals.execution_providers:
        return 1
    if 'ROCMExecutionProvider' in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # prevent tensorflow memory leak
    gpus = tensorflow.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tensorflow.config.experimental.set_memory_growth(gpu, True)
    # limit memory usage
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024 ** 3
        if platform.system().lower() == 'darwin':
            memory = modules.globals.max_memory * 1024 ** 6
        if platform.system().lower() == 'windows':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetProcessWorkingSetSize(-1, ctypes.c_size_t(memory), ctypes.c_size_t(memory))
        else:
            import resource
            resource.setrlimit(resource.RLIMIT_DATA, (memory, memory))


def release_resources() -> None:
    if 'CUDAExecutionProvider' in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status('Python version is not supported - please upgrade to 3.9 or higher.')
        return False
    if not shutil.which('ffmpeg'):
        update_status('ffmpeg is not installed.')
        return False
    return True


def update_status(message: str, scope: str = 'DLC.CORE') -> None:
    print(f'[{scope}] {message}')
    if (not modules.globals.headless) and getattr(modules.globals, 'ui_ready', False):
        ui.update_status(message)

def start() -> None:
    # Ensure Nano Banana preprocessing on source right before processing
    try:
        if getattr(modules.globals, 'enable_nano_banana', False):
            from modules.nano_banana import preprocess_image_with_gemini
            # Simple mode: preprocess source file path
            if not getattr(modules.globals, 'map_faces', False):
                if modules.globals.source_path and is_image(modules.globals.source_path):
                    update_status('Nano Banana: preprocessing source image (pre-start)...')
                    try:
                        new_src = preprocess_image_with_gemini(
                            modules.globals.source_path,
                            modules.globals.nano_banana_prompt,
                            modules.globals.nano_banana_model,
                        )
                        if new_src:
                            update_status(f"Nano Banana applied to source → {new_src}")
                            modules.globals.source_path = new_src
                    except Exception as e:
                        update_status(f"Nano Banana: pre-start failed: {e}")
            else:
                # Map mode: preprocess in-memory source crops in souce_target_map
                maps = getattr(modules.globals, 'souce_target_map', []) or []
                total = sum(1 for it in maps if isinstance(it, dict) and it.get('source') and it['source'].get('cv2') is not None)
                done = 0
                if total:
                    update_status(f'Nano Banana: preprocessing {total} mapped source(s) (pre-start)...')
                for item in maps:
                    try:
                        src = item.get('source') if isinstance(item, dict) else None
                        if not src:
                            continue
                        src_path = src.get('path')
                        nb = None
                        if src_path and os.path.exists(src_path) and is_image(src_path):
                            # Prefer full image path if available
                            nb = preprocess_image_with_gemini(
                                src_path,
                                modules.globals.nano_banana_prompt,
                                modules.globals.nano_banana_model,
                            )
                        else:
                            # Fallback to processing the cropped face with padding
                            crop = src.get('cv2')
                            if crop is None:
                                continue
                            h, w = crop.shape[:2]
                            pad = max(16, int(min(h, w) * 0.15))
                            crop_padded = cv2.copyMakeBorder(crop, pad, pad, pad, pad, cv2.BORDER_REFLECT)
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp:
                                tmp_path = tmp.name
                            try:
                                cv2.imwrite(tmp_path, crop_padded)
                                nb = preprocess_image_with_gemini(
                                    tmp_path,
                                    modules.globals.nano_banana_prompt,
                                    modules.globals.nano_banana_model,
                                )
                            finally:
                                try:
                                    os.remove(tmp_path)
                                except Exception:
                                    pass
                        if nb:
                            cv_nb = cv2.imread(nb)
                            if cv_nb is None:
                                # Fallback: load via PIL then convert to BGR
                                try:
                                    from PIL import Image
                                    import numpy as np
                                    pil_img = Image.open(nb).convert('RGB')
                                    cv_nb = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                                except Exception as e:
                                    print(f"[NanoBanana][core_map] Failed to read nb image: {e}")
                                    cv_nb = None
                            if cv_nb is not None:
                                # If very small, upscale to aid detection
                                mh, mw = cv_nb.shape[:2]
                                if min(mh, mw) < 256:
                                    scale = max(1.0, 256.0 / float(min(mh, mw)))
                                    cv_nb = cv2.resize(cv_nb, (int(mw*scale), int(mh*scale)))
                                face_nb = get_one_face(cv_nb)
                                if face_nb and getattr(face_nb, 'bbox', None) is not None:
                                    x_min, y_min, x_max, y_max = face_nb["bbox"]
                                    item["source"]["cv2"] = cv_nb[int(y_min): int(y_max), int(x_min): int(x_max)]
                                    item["source"]["face"] = face_nb
                                    done += 1
                                else:
                                    # No face re-detected; keep original face but replace crop
                                    item["source"]["cv2"] = cv_nb
                                    done += 1
                            else:
                                print("[NanoBanana][core_map] nb path unreadable; skipping this item")
                        else:
                            print("[NanoBanana][core_map] preprocess returned None; skipping this item")
                    except Exception as e:
                        print(f"[NanoBanana][core_map] Exception: {e}")
                if total:
                    update_status(f'Nano Banana: completed {done}/{total} mapped source(s)')
    except Exception:
        pass
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_start():
            return
    update_status('Processing...')
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
            update_status('Progressing...', frame_processor.NAME)
            frame_processor.process_image(modules.globals.source_path, modules.globals.output_path, modules.globals.output_path)
            release_resources()
        if is_image(modules.globals.target_path):
            update_status('Processing to image succeed!')
        else:
            update_status('Processing to image failed!')
        return
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(modules.globals.target_path, destroy):
        return

    if not modules.globals.map_faces:
        update_status('Creating temp resources...')
        create_temp(modules.globals.target_path)
        update_status('Extracting frames...')
        extract_frames(modules.globals.target_path)

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        update_status('Progressing...', frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        release_resources()
    # handles fps
    if modules.globals.keep_fps:
        update_status('Detecting fps...')
        fps = detect_fps(modules.globals.target_path)
        update_status(f'Creating video with {fps} fps...')
        create_video(modules.globals.target_path, fps)
    else:
        update_status('Creating video with 30.0 fps...')
        create_video(modules.globals.target_path)
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status('Restoring audio...')
        else:
            update_status('Restoring audio might cause issues as fps are not kept...')
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # clean and validate
    clean_temp(modules.globals.target_path)
    if is_video(modules.globals.target_path):
        update_status('Processing to video succeed!')
    else:
        update_status('Processing to video failed!')


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit: quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    # Load .env for GEMINI_API_KEY if present
    try:
        load_dotenv()
    except Exception:
        pass
    # If GEMINI_API_KEY is present, auto-enable nano banana unless explicitly disabled
    try:
        if os.environ.get('GEMINI_API_KEY') and not getattr(modules.globals, 'enable_nano_banana', False) and not getattr(modules.globals, 'nano_banana_user_override', False):
            modules.globals.enable_nano_banana = True
            update_status('Nano Banana enabled (GEMINI_API_KEY detected)')
    except Exception:
        pass
    for frame_processor in get_frame_processors_modules(modules.globals.frame_processors):
        if not frame_processor.pre_check():
            return
    limit_resources()
    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()
