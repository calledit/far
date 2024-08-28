import pytorch_lightning as pl
import argparse

from src.config.default import get_cfg_defaults

from src.lightning.lightning_loftr import PL_LoFTR
import sys
import torch
from pathlib import Path
import json
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
sys.path.append('third_party/prior_ransac')

def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default="", help='path to the checkpoint')
    parser.add_argument(
        '--exp_name', type=str, default="", help='exp_name, if not present use ckpt')
    parser.add_argument(
        '--batch_size', type=int, default=1, help='batch_size per gpu')
    parser.add_argument(
        '--output', type=str, default='tranforms.json', help='output file location')
    parser.add_argument(
        '--frame_image_folder', type=str, default="/root/far/mp3d_loftr/data/imgs_4fps/", help='folder with video frames')
    parser.add_argument(
        '--mask_image_folder', type=str, default="/root/far/mp3d_loftr/data/masks_4fps/", help='folder with video mask frames')
    parser.add_argument(
        '--num_workers', type=int, default=2)
    parser.add_argument(
        '--fx', type=float, default=517.97, help='focal length x')
    parser.add_argument(
        '--fy', type=float, default=517.97, help='focal length y')
    parser.add_argument(
        '--cx', type=float, default=320, help='principal point x')
    parser.add_argument(
        '--cy', type=float, default=240, help='principal point y')
    parser.add_argument(
        '--h', type=int, default=640, help='img height x')
    parser.add_argument(
        '--w', type=int, default=480, help='img width y')
    parser.add_argument(
        '--img_path0', type=str, default="test", help='img path 0')
    parser.add_argument(
        '--img_path1', type=str, default="test", help='img path 1')
    parser = pl.Trainer.add_argparse_args(parser)
    return parser.parse_args()

if __name__ == '__main__':
    # parse arguments
    args = parse_args()

    # init default-cfg and merge it with the main- and data-cfg
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    pl.seed_everything(config.TRAINER.SEED)  # reproducibility

    # not relevant
    config.LOFTR.FEAT_SIZE = 128
    config.LOFTR.NUM_ENCODER_LAYERS = 6
    config.LOFTR.NUM_BANDS = 10
    config.LOFTR.CORRESPONDENCE_TRANSFORMER_USE_POS_ENCODING = False
    config.LOFTR.CORRESPONDENCE_TF_USE_GLOBAL_AVG_POOLING = False
    config.LOFTR.CORRESPONDENCE_TF_USE_FEATS = False
    config.LOFTR.CTF_CAT_FEATS = False
    config.LOFTR.NUM_HEADS = 8
    config.USE_PRED_CORR = False
    config.STRICT_FALSE = False
    config.EVAL_FIT_ONLY = False


    # relevant
    config.LOFTR.PREDICT_TRANSLATION_SCALE = False
    config.LOFTR.REGRESS_RT = True
    config.LOFTR.REGRESS_LOFTR_LAYERS = 1
    config.LOFTR.REGRESS.USE_POS_EMBEDDING = True
    config.LOFTR.REGRESS.REGRESS_USE_NUM_CORRES = True
    config.LOFTR.COARSE.LAYER_NAMES = ['self', 'cross'] * 3

    # pass into config
    config.LOFTR.FROM_SAVED_PREDS = None
    config.LOFTR.SOLVER = "prior_ransac"
    config.LOFTR.USE_MANY_RANSAC_THR = True

    config.LOFTR.FINE_PRED_STEPS = 2
    config.LOFTR.REGRESS.SAVE_MLP_FEATS = False
    config.LOFTR.REGRESS.USE_SIMPLE_MOE = True
    config.LOFTR.REGRESS.USE_2WT = True
    config.LOFTR.REGRESS.USE_5050_WEIGHT = False
    config.LOFTR.REGRESS.USE_1WT = False
    config.LOFTR.REGRESS.SCALE_8PT = True
    config.LOFTR.REGRESS.SAVE_GATING_WEIGHTS = False
    config.LOFTR.TRAINING = False

    config.LOAD_PREDICTIONS_PATH = None
    config.EXP_NAME = args.exp_name
    config.USE_CORRESPONDENCE_TRANSFORMER = False
    config.CORRESPONDENCES_USE_FIT_ONLY = False
    config.EVAL_SPLIT = "test"
    config.PL_VERSION = pl.__version__
    def get_intrinsics(fx, fy, cx, cy):
        K = [[fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]]
        K = np.array(K)
        K = torch.from_numpy(K.astype(np.double)).unsqueeze(0).cuda()
        return K, K


    #fov_x, fov_y = 47.5, 35.5
    def intrin_from(fov_x, fov_y):
        fx = 640 / (2 * np.tan(np.deg2rad(fov_x) / 2))
        fy = 480 / (2 * np.tan(np.deg2rad(fov_y) / 2))
        return get_intrinsics(fx, fy, float(args.cx), float(args.cy))

    #K_0, K_1 = get_intrinsics(float(args.fx), float(args.fy), float(args.cx), float(args.cy))
    #K_0, K_1 = intrin_from(61.9, 43.6)

    def default(obj):
        if type(obj).__module__ == np.__name__:
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
               return obj.item()
        raise TypeError('Unknown type:', type(obj))


    # lightning module
    model = PL_LoFTR(config, pretrained_ckpt=args.ckpt_path, split="test").eval().cuda()

    # unused
    depth0 = depth1 = torch.tensor([]).unsqueeze(0).cuda()
    T_0to1 = T_1to0 = torch.tensor([]).unsqueeze(0).cuda()
    scene_name = torch.tensor([]).unsqueeze(0).cuda()
    loaded_preds = torch.tensor([]).unsqueeze(0).cuda()
    lightweight_numcorr = torch.tensor([0]).unsqueeze(0).cuda()

    def load_image(path, is_mask = False, is_depth = False):
        # load data
        image0 = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        image0 = cv2.resize(image0, (args.w, args.h))
        deepth_image_offset = 0.7
        deepth_image_scale = 5
        if is_depth:
            image0 = (torch.from_numpy(image0).double()/(255*deepth_image_scale)) + deepth_image_offset
        elif is_mask:
            _, image0 = cv2.threshold(image0, 10, 255, cv2.THRESH_BINARY)
            image0 = torch.from_numpy(image0).float().cuda() / 255
            image0 = image0.unsqueeze(0)
        else:
            image0 = torch.from_numpy(image0).float()[None].unsqueeze(0).cuda() / 255
        return image0

    def solve_tranform(fov_x, fov_y, image0, image1, expect_fail = False, mask0 = None, mask1 = None):
        K_0, K_1 = intrin_from(fov_x, fov_y)
        batch = {
            'image0': image0,   # (1, h, w)
            'image1': image1,
            'K0': K_0,  # (3, 3)
            'K1': K_1,
            # below is unused
            'depth0': depth0,   # (h, w)
            'depth1': depth1,
            'T_0to1': T_0to1,   # (4, 4)
            'T_1to0': T_1to0,
            'dataset_name': ['mp3d'],
            'scene_id': scene_name,
            'pair_id': 0,
            'pair_names': (args.img_path0, args.img_path1),
            'loaded_predictions': loaded_preds,
            'lightweight_numcorr': lightweight_numcorr,
            'expect_fail': expect_fail
        }

        if mask0 is not None:
            batch['mask0'] = mask0
        if mask1 is not None:
            batch['mask1'] = mask1


        # forward pass
        batch = model.test_step(batch, batch_idx=0, skip_eval=True)


        ret = {}
        ret['transform'] = batch['loftr_rt'].cpu().numpy()
        ret['transform2'] = batch['priorRT']
        ret['num_correspondences'] = batch['num_correspondences'].cpu().numpy()[0]
        #ret['translation_scale'] = batch['translation_scale'].cpu().numpy()

        #if (np.eye(4)[:3,:4] == ret['transform']).all():#Sometimes an eye seams to mean failure somethimes that the camera has not moved
        #    return False
        return ret

    def solve_tranform_no_fov(image0, image1, nr_of_tries=5, initial_gues_fov_x = 47.5, initial_gues_fov_y = 35.5, mask0 = None, mask1 = None):
        try_fov_x, try_fov_y, solution = find_fov(
                [(image0, image1)],
                initial_gues_fov_x = initial_gues_fov_x,
                initial_gues_fov_y = initial_gues_fov_y,
                x_steps = nr_of_tries,
                y_steps = nr_of_tries,
                mask0 = mask0,
                mask1 = mask1,
                return_when_found = True)
        return (try_fov_x, try_fov_y, solution)

    def find_fov(loaded_image_pairs = [], return_when_found=False, initial_gues_fov_x = 47.5, initial_gues_fov_y = 35.5, step=1, x_steps = 15, y_steps = 10, mask0 = None, mask1 = None):
        fov_x_2_try = [initial_gues_fov_x]
        fov_y_2_try = [initial_gues_fov_y]

        x = 1
        while x < x_steps:
            fov_x_2_try.append(initial_gues_fov_x - x)
            fov_x_2_try.append(initial_gues_fov_x + x)
            x += step

        y = 1
        while y < y_steps:
            fov_y_2_try.append(initial_gues_fov_y - y)
            fov_y_2_try.append(initial_gues_fov_y + y)
            y += step

        tries = {}
        for try_fov_y in sorted(fov_y_2_try):
            tries[try_fov_y] = {}
            for try_fov_x  in sorted(fov_x_2_try):
                tries[try_fov_y][try_fov_x] = 0

        if not return_when_found:
            fovs = open("fov_solutions.json", "w")
        solutions = []
        for try_fov_y in fov_y_2_try:
            for try_fov_x in fov_x_2_try:
                #print("trying: ", "fov_x:", try_fov_x, "fov_y:", try_fov_y)
                had_failure = False
                solution = False
                for earlier, later in loaded_image_pairs:
                    solution = solve_tranform(try_fov_x, try_fov_y, later, earlier, expect_fail = not return_when_found, mask0 = mask0, mask1 = mask1)
                    if not (solution is not False):
                        had_failure = True
                        if return_when_found:
                            break
                    else:
                        tries[try_fov_y][try_fov_x] +=1
                if not had_failure:
                    if return_when_found:
                        return (try_fov_x, try_fov_y, solution)
                    sol = [try_fov_x, try_fov_y]
                    solutions.append(sol)
                    print("found solution fov in degres:", sol)

        if not return_when_found:
            fovs.write(json.dumps(tries, default=default))
            fovs.close()
            x_solutions = {}
            y_solutions = {}
            for tra in solutions:
                if tra[0] not in x_solutions:
                    x_solutions[tra[0]] = 0
                x_solutions[tra[0]] += 1

                if tra[1] not in y_solutions:
                    y_solutions[tra[1]] = 0
                y_solutions[tra[1]] += 1
            nr_tries = len(fov_y_2_try)*len(fov_x_2_try)
            for x in x_solutions:
                x_solutions[x] /=nr_tries
            for y in y_solutions:
                y_solutions[y] /=nr_tries

            x_solutions = {k: v for k, v in sorted(x_solutions.items(), key=lambda item: item[1])}
            y_solutions = {k: v for k, v in sorted(y_solutions.items(), key=lambda item: item[1])}
            print(x_solutions)
            print(y_solutions)
        return (False, False, False)

    #search for instrinsics
    search_solution_to_list_of_frames = False
    if search_solution_to_list_of_frames:
        frame2frame = [13, 15, 17, 34]
        loaded_image_pairs = []
        for i in frame2frame:
            loaded_image_pairs.append((load_image(f"{args.frame_image_folder}{(i-1):06d}.png"), load_image(f"{args.frame_image_folder}{i:06d}.png")))

        find_fov(
            loaded_image_pairs=loaded_image_pairs,
            initial_gues_fov_x = 56,
            initial_gues_fov_y = 32,
            step=1, x_steps = 10, y_steps = 10
            )



    #args.frame_image_folder = "/root/far/mp3d_loftr/data/imgs_4fps/"
    images = [f for f in listdir(args.frame_image_folder) if isfile(join(args.frame_image_folder, f))]

    first = True
    trans = open(args.output, "w")
    trans.write("[")
    nths = [1000, 100, 50, 25, 10, 5, 1]
    #nths = [1]
    for nth in nths:
        last_frame, last_img_name = None, None
        num = -1
        for img_name in images:
            num+=1
            if not num%nth == 0:
                continue
            img_path = join(args.frame_image_folder, img_name)
            mask_path = join(args.mask_image_folder, img_name)
            curent_frame = load_image(img_path)
            curent_mask_frame = load_image(mask_path, True)
            if last_frame == None:
                last_frame = curent_frame
                last_mask_frame = curent_mask_frame
                last_img_name = img_name
                continue

            if first:
                first = False
            else:
                trans.write(",")#Write json comma

            fov_x, fov_y = 60, 46.82
            json_line = solve_tranform(fov_x, fov_y, curent_frame, last_frame, mask0 = curent_mask_frame, mask1 = last_mask_frame)

            json_line['nth'] = nth
            json_line['from_frame'] =int(Path(last_img_name).stem)
            json_line['to_frame'] = int(Path(img_name).stem)

            #if not (transform is not False):
            #    solv_x, solv_y, transform = solve_tranform_no_fov(curent_frame, last_frame, nr_of_tries=2, initial_gues_fov_x = fov_x, initial_gues_fov_y = fov_y)
            #    if transform is not False:
            #        json_line['fov_x'] = solv_x
            #        json_line['fov_y'] = solv_y

            dumped = json.dumps(json_line, default=default)

            print(dumped)
            trans.write(dumped+"\n")

            # output
            #print("predicted pose is:\n", np.round(batch['loftr_rt'].cpu().numpy(),4))
            last_frame = curent_frame
            last_mask_frame = curent_mask_frame
            last_img_name = img_name
    trans.write("]")
