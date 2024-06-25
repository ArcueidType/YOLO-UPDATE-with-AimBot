import pygetwindow
import bettercam
from config import *


def get_cs2_window() -> (bettercam.BetterCam, int, int):
    try:
        windows = pygetwindow.getAllWindows()
        cs2_window = None
        for window in windows:
            if window.title == 'Counter-Strike 2':
                cs2_window = window
                break

        if cs2_window is None:
            print('Counter-Strike 2 is not running')
            return None, None, None

    except Exception as e:
        print('Failed to get CS2 window: {}'.format(e))
        return None, None, None

    retry_times = 10
    is_activate = False
    while retry_times > 0:
        try:
            cs2_window.activate()
            is_activate = True
            break
        except pygetwindow.PyGetWindowException as gwe:
            print('Failed to activate CS2 window: {}'.format(gwe))
            print('Retrying')
        except Exception as e:
            print('Failed to activate CS2 window: {}'.format(e))
            is_activate = False
            retry_times = 0
        retry_times -= 1

    if not is_activate:
        return None, None, None

    print('CS2 window activated')

    scan_region_left = ((cs2_window.left + cs2_window.right) // 2) - (SCAN_REGION_WIDTH // 2)
    scan_region_top = cs2_window.top + ((cs2_window.height - SCAN_REGION_HEIGHT) // 2)
    scan_region_right = scan_region_left + SCAN_REGION_WIDTH
    scan_region_bottom = scan_region_top + SCAN_REGION_HEIGHT

    scan_region = (scan_region_left, scan_region_top, scan_region_right, scan_region_bottom)

    center_width = SCAN_REGION_WIDTH // 2
    center_height = SCAN_REGION_HEIGHT // 2

    camera = bettercam.create(region=scan_region, output_color='BGRA', max_buffer_len=512)
    if camera is None:
        print('Failed to create camera')
        return None, None, None

    camera.start(target_fps=120, video_mode=True)

    return camera, center_width, center_height
