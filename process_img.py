#   rmBlackBorder: remove the black borders of one image
#   return: cropped image
def rmBlackBorder(
    src,  # input image
    thres,  # threshold for cropping: sum([r,g,b] - [0,0,0](black))
    diff,  # max tolerable difference between black borders on two side
    shrink  # number of pixels to shrink after the blackBorders removed
):
    #
    #   remove the black border on both right and left side
    #
    nRow = src.shape[0]
    nCol = src.shape[1]
    left = -1
    right = nCol

    for i in [0, nRow // 2, nRow - 1]:
        curLeft = -1
        curRight = nCol
        for j in range(0, nCol - 1):
            if (sum(list(src[i, j])) <= thres and curLeft == j - 1):
                curLeft += 1
        if left == -1:
            left = curLeft
        if curLeft < left:
            left = curLeft

        for j in range(nCol - 1, 0, -1):
            if (sum(list(src[i, j])) <= thres and curRight == j + 1):
                curRight -= 1
        if right == nCol:
            right = curRight
        if curRight > right:
            right = curRight

    if min(left, right) >= 1 \
        and abs((left + 1) - (nCol - right)) <= diff \
        and right - left > 0\
        and (right - 1 - shrink) > (left + 1 + shrink):
        # print('left margin: %d\n' % left)
        # print('right margin: %d\n' % right)


        src = src[0: nRow - 1, left + 1 + shrink: right - 1 - shrink, :]
    else:
        src = src

    #
    #   remove the black border on both up and down side
    #
    nRow = src.shape[0]
    nCol = src.shape[1]
    up = -1
    down = nRow

    for j in [0, nCol // 2, nCol - 1]:
        curUp = -1
        curDown = nRow
        for i in range(0, nRow - 1):
            if (sum(list(src[i, j])) <= thres and curUp == i - 1):
                curUp += 1
        if up == -1:
            up = curUp
        if curUp < up:
            up = curUp

        for i in range(nRow - 1, 0, -1):
            if (sum(list(src[i, j])) <= thres and curDown == i + 1):
                curDown -= 1
        if down == nRow:
            down = curDown
        if curDown > down:
            down = curDown

    if min(up, down) >= 1 and abs((up + 1) - (nRow - down)) <= diff and down - up > 0:
        # print
        # 'up margin: %d\n' % up
        # print
        # 'down margin: %d\n' % down

        dst = src[up + 1 + shrink: down - 1 - shrink, 0: nCol - 1, :]
        res = [True,up,shrink,down,nCol]
    else:
        dst = src
        res = [False]

    return res
import cv2
import os
from concurrent.futures import ProcessPoolExecutor

EXCLUDE=['v_ApplyEyeMakeup_g07_c05', 'v_ApplyEyeMakeup_g23_c04','v_ApplyLipstick_g19_c02',
         'v_BlowDryHair_g12_c05', 'v_BlowingCandles_g19_c01','v_Bowling_g03_c02',
         'v_Bowling_g03_c06','v_Bowling_g08_c04', 'v_Bowling_g09_c06','v_Bowling_g18_c02',
         'v_Bowling_g18_c03','v_BrushingTeeth_g20_c01','v_BrushingTeeth_g20_c02','v_BrushingTeeth_g20_c04',
         'v_CleanAndJerk_g02_c03','v_CliffDiving_g09_c05','v_CliffDiving_g16_c02','v_CricketBowling_g16_c02',
         'v_CricketShot_g24_c01','v_CricketShot_g24_c02','v_CricketShot_g24_c03',
         'v_CricketShot_g24_c07','v_CuttingInKitchen_g10_c03','v_CuttingInKitchen_g10_c04',
         'v_CuttingInKitchen_g13_c01','v_CuttingInKitchen_g13_c02',
         'v_Drumming_g13_c07','v_Drumming_g19_c02','v_Drumming_g19_c06',
         'v_Drumming_g23_c01','v_Drumming_g23_c03','v_Drumming_g23_c05',
         'v_FieldHockeyPenalty_g01_c02','v_FieldHockeyPenalty_g01_c04',
         'v_Haircut_g24_c05','v_HeadMassage_g07_c04', 'v_HorseRiding_g15_c04',
         'v_HulaHoop_g25_c01','v_HulaHoop_g25_c05','v_HulaHoop_g25_c03','v_HulaHoop_g25_c04',
         'v_JumpRope_g18_c01','v_JumpRope_g18_c02','v_JumpRope_g18_c03','v_JumpRope_g18_c05',
         'v_MilitaryParade_g17_c02','v_MilitaryParade_g17_c02','v_Mixing_g06_c03','v_Mixing_g06_c05',
         'v_MoppingFloor_g25_c01','v_PlayingCello_g08_c01','v_PlayingCello_g08_c04','v_PlayingDhol_g03_c03',
         'v_PlayingFlute_g05_c03', 'v_PlayingViolin_g10_c03','v_PlayingViolin_g21_c01',
         'v_PullUps_g18_c03','v_PullUps_g18_c04','v_Punch_g10_c02','v_Rafting_g13_c04','v_Rafting_g13_c05',
         'v_ShavingBeard_g09_c01','v_ShavingBeard_g09_c02','v_ShavingBeard_g09_c03','v_ShavingBeard_g09_c04',
         'v_ShavingBeard_g09_c05','v_ShavingBeard_g09_c06','v_ShavingBeard_g09_c07',
         'v_Shotput_g07_c02','v_Shotput_g18_c02','v_Shotput_g18_c03','v_Shotput_g18_c04',
         'v_Skiing_g11_c02','v_Skiing_g11_c04','v_SkyDiving_g13_c01','v_SkyDiving_g13_c02',
         'v_SkyDiving_g13_c03','v_SkyDiving_g13_c04','v_SumoWrestling_g18_c05',
         'v_TableTennisShot_g20_c05','v_ThrowDiscus_g03_c01','v_ThrowDiscus_g21_c06',
         'v_TrampolineJumping_g01_c03','v_TrampolineJumping_g19_c02','v_WallPushups_g02_c03',
         'v_WallPushups_g12_c01','v_WallPushups_g12_c02','v_WallPushups_g12_c03','v_WallPushups_g12_c04',
         'v_WallPushups_g12_c05','v_WallPushups_g16_c01','v_WallPushups_g16_c02','v_WallPushups_g16_c03',
         'v_WallPushups_g16_c04','v_WallPushups_g16_c05','v_WallPushups_g16_c06','v_WallPushups_g16_c07',
         'v_YoYo_g20_c01']

def run(c):
    cls_path = os.path.join(jpg_path, c)
    avi_dirs = os.listdir(cls_path)
    avi_dirs = [c for c in avi_dirs if not c.startswith('.')]
    for avi in avi_dirs:
        if avi in EXCLUDE:
            continue
        # idx += 1
        # print(idx)
        avi_path = os.path.join(cls_path, avi)
        jpgs = os.listdir(avi_path)
        jpgs = [c for c in jpgs if not c.startswith('.')]
        for i, jpg in enumerate(jpgs):
            img_path = os.path.join(avi_path, jpg)
            img = cv2.imread(img_path)
            if i == 0:
                res = rmBlackBorder(img, 10, 100, 2)
                if res[0]:
                    print(avi_path)
                origin_size = (img.shape[1], img.shape[0])
            if res[0]:
                up, shrink, down, nCol = res[1:]
                img = img[up + 1 + shrink: down - 1 - shrink, 0: nCol - 1, :]
                img = cv2.resize(img, (origin_size))
                cv2.imwrite(img_path, img)
                if i == 0:
                    cv2.imwrite(os.path.join(inspect_path, avi + '.jpg'), img)

if __name__ == '__main__':
    inspect_path = '/Users/alex/baidu/3dresnet-data/inspect-ucf-101'
    if not os.path.exists(inspect_path):
        os.makedirs(inspect_path)
    jpg_path = '/Users/alex/baidu/3dresnet-data/UCF-101-jpg'

    cls_dir = os.listdir(jpg_path)
    cls_dir = [c for c in cls_dir if not c.startswith('.')]
    idx=0
    exector = ProcessPoolExecutor(5)
    for _ in exector.map(run, cls_dir):
        pass


