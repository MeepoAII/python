import cv2


class CoorTransfer:
    def crop_map(self, start_x, start_y, x, y):
        return x+start_x, y+start_y


def main():
    a = CoorTransfer()
    x, y = a.crop_map(126, 263, 525, 460)
    print(x, y)

    return


if __name__ == '__main__':
    main()
