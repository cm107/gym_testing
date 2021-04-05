import cv2
from tqdm import tqdm
from common_utils.path_utils import get_dirnames_in_dir, get_all_files_of_extension, \
    get_rootname_from_path, get_filename
from common_utils.file_utils import make_dir_if_not_exists
from common_utils.cv_drawing_utils import draw_text_rows_in_corner
from common_utils.image_utils import collage_from_img_buffer
from streamer.streamer import StreamerList
from streamer.recorder.stream_writer import StreamWriter

src_root_dir = 'output-ppo_comparison'
dst_root_dir = f'{src_root_dir}/comparison'
make_dir_if_not_exists(dst_root_dir)
env_names = [dirname for dirname in get_dirnames_in_dir(src_root_dir) if dirname != 'comparison' and 'ram' in dirname]
for env_name in env_names:
    src_env_dir = f'{src_root_dir}/{env_name}'
    dst_env_dir = f'{dst_root_dir}/{env_name}'
    make_dir_if_not_exists(dst_env_dir)
    # run_id_list = get_dirnames_in_dir(src_env_dir)
    run_id_list = [
        'gamma0.99-tau0.95-no_gae-no_ppo',
        'gamma0.99-tau0.95-gae-no_ppo',
        'gamma0.99-tau0.95-no_gae-ppo',
        'gamma0.99-tau0.95-gae-ppo'
    ]
    img_filenames = [get_filename(path) for path in get_all_files_of_extension(f'{src_env_dir}/{run_id_list[0]}', extension='png')]
    for img_filename in img_filenames:
        img_buffer = []

        for run_id in run_id_list:
            src_run_id_dir = f"{src_env_dir}/{run_id}"
            src_img_path = f'{src_run_id_dir}/{img_filename}'
            img = cv2.imread(src_img_path)
            img = draw_text_rows_in_corner(
                img=img,
                row_text_list=[
                    f"Env: {env_name}",
                    f"Run ID: {run_id}"
                ],
                row_height=img.shape[0]*0.03,
                thickness=1,
                color=[255, 0, 0],
                corner='topleft'
            )
            img_buffer.append(img)
        comparison_img = collage_from_img_buffer(img_buffer, collage_shape=(2,2))
        dst_img_path = f"{dst_env_dir}/{img_filename}"
        cv2.imwrite(dst_img_path, comparison_img)
    
    for video_filename in ['infer.avi']:
        streamer_list = StreamerList([f'{src_env_dir}/{run_id}/{video_filename}' for run_id in run_id_list])
        stream_writer = StreamWriter(
            video_save_path=f'{dst_env_dir}/{video_filename}',
            fps=streamer_list.get_fps()
        )

        pbar = tqdm(total=streamer_list.get_frame_count(), unit='frame(s)', leave=True)
        pbar.set_description('Writing Comparison Videos')
        while streamer_list.is_open() and streamer_list.is_playing():
            img_buffer = streamer_list.get_frame()
            comparison_img = collage_from_img_buffer(img_buffer, collage_shape=(2,2))
            stream_writer.step(comparison_img)
            pbar.update()
        streamer_list.close()
        stream_writer.close()
        pbar.close()