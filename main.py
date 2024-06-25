import cv2

import CS2Window
from ultralytics import YOLO
import torch
import numpy as np
import pandas as pd
import win32api
import win32con
from einops import rearrange, repeat
from config import *


def aim_target(targets, center_width, center_height):
    if len(targets) > 0:
        # targets['dist_from_center'] = np.sqrt(
        #     (targets['center_x'] - center_width)**2 + (targets['center_y'] - center_height)**2
        # )
        #
        # targets = targets.sort_values(by='dist_from_center', ascending=True)

        x = targets.iloc[0]['center_x']
        y = targets.iloc[0]['center_y'] * 0.96

        move_x = x - center_width
        move_y = y - center_height

        if win32api.GetKeyState(0x14):
            win32api.mouse_event(
                win32con.MOUSEEVENTF_MOVE, int(move_x * MOUSE_MOVE_RATE), int(move_y * MOUSE_MOVE_RATE)
                , 0, 0)


def mode_all(ct_targets_head, t_targets_head, ct_targets_body, t_targets_body, center_width, center_height):
    if len(ct_targets_head) > 0 and HEAD_SHOT_MODE:
        aim_target(ct_targets_head, center_width, center_height)
    elif len(t_targets_head) > 0 and HEAD_SHOT_MODE:
        aim_target(t_targets_head, center_width, center_height)
    elif len(ct_targets_body) > 0 and not HEAD_SHOT_MODE:
        aim_target(ct_targets_body, center_width, center_height)
    elif len(t_targets_body) > 0 and not HEAD_SHOT_MODE:
        aim_target(t_targets_body, center_width, center_height)


def mode_ct(ct_targets_head, ct_targets_body, center_width, center_height):
    if len(ct_targets_head) > 0 and HEAD_SHOT_MODE:
        aim_target(ct_targets_head, center_width, center_height)
    elif len(ct_targets_body) > 0 and not HEAD_SHOT_MODE:
        aim_target(ct_targets_body, center_width, center_height)


def mode_t(t_targets_head, t_targets_body, center_width, center_height):
    if len(t_targets_head) > 0 and HEAD_SHOT_MODE:
        aim_target(t_targets_head, center_width, center_height)
    elif len(t_targets_body) > 0 and not HEAD_SHOT_MODE:
        aim_target(t_targets_body, center_width, center_height)


def draw_targets(img, targets, label):
    for i in range(len(targets)):
        width_half = targets['width'][i] / 2
        height_half = targets['height'][i] / 2
        center_x = targets['center_x'][i]
        center_y = targets['center_y'][i]
        start_x, start_y = int(center_x - width_half), int(center_y - height_half)
        end_x, end_y = int(center_x + width_half), int(center_y + height_half)

        cv2.rectangle(img, (start_x, start_y), (end_x, end_y), (255, 0, 0), 2)
        text_x = start_x
        text_y = start_y - 15 if start_y - 15 > 15 else start_y + 15
        cv2.putText(
            img, label, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (255, 255, 255), 2
        )

    return img


def main():
    camera, center_width, center_height = CS2Window.get_cs2_window()

    model = YOLO(model=MODEL)

    with torch.no_grad():
        while win32api.GetAsyncKeyState(ord(QUIT_KEY)) == 0:
            img_arr = np.array(camera.get_latest_frame())
            img = torch.from_numpy(img_arr)

            if img.shape[2] == 4:
                img = img[:, :, :3]

            img = rearrange(repeat(img, 'h w c -> b h w c', b=1), 'b h w c -> b c h w')
            if torch.cuda.is_available():
                img = img.half()
                img /= 255.0

            results = model(img, conf=CONFIDENCE_THRESHOLD)

            ct_targets_head = []
            ct_targets_body = []
            t_targets_head = []
            t_targets_body = []

            for result in results:
                for i, cls in enumerate(result.boxes.cls):
                    # ct_targets_head.append(result.boxes.xywh[i])
                    if cls == 1:
                        ct_targets_head.append(result.boxes.xywh[i].cpu().numpy())
                    elif cls == 4:
                        t_targets_head.append(result.boxes.xywh[i].cpu().numpy())
                    elif cls == 0:
                        ct_targets_body.append(result.boxes.xywh[i].cpu().numpy())
                    elif cls == 3:
                        t_targets_body.append(result.boxes.xywh[i].cpu().numpy())
                    else:
                        continue

            ct_targets_head = pd.DataFrame(ct_targets_head, columns=['center_x', 'center_y', 'width', 'height'])
            t_targets_head = pd.DataFrame(t_targets_head, columns=['center_x', 'center_y', 'width', 'height'])
            ct_targets_body = pd.DataFrame(ct_targets_body, columns=['center_x', 'center_y', 'width', 'height'])
            t_targets_body = pd.DataFrame(t_targets_body, columns=['center_x', 'center_y', 'width', 'height'])

            if MODE == 0:
                mode_all(ct_targets_head, t_targets_head, ct_targets_body, t_targets_body, center_width, center_height)
            elif MODE == 1:
                mode_ct(ct_targets_head, ct_targets_body, center_width, center_height)
            elif MODE == 2:
                mode_t(t_targets_head, t_targets_body, center_width, center_height)
            else:
                print('Unknown mode')
                exit()

            if VISUAL:
                draw_targets(img_arr, ct_targets_head, 'CT-HEAD')
                draw_targets(img_arr, ct_targets_body, 'CT-BODY')
                draw_targets(img_arr, t_targets_head, 'T-HEAD')
                draw_targets(img_arr, t_targets_body, 'T-BODY')

                cv2.imshow('Current Model View', img_arr)
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    exit()

        camera.stop()


if __name__ == '__main__':
    main()
