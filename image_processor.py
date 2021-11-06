import cv2
import numpy as np


class ImageProcessor:

    def __init__(self):
        # character width threshold
        self.width_thr = 40

    def _remove_background(self, img):
        """
        Removes background color from image.
        """
        for j in range(self.col):
            if img[self.row-1, j] <= img[0, j]:
                rgb = img[0, j]
            elif ((int(img[self.row-1, j]) - int(img[0, j])) /
                  img[self.row-1, j] < 0.06):
                rgb = img[0, j]
            else:
                rgb = round(img[self.row-1, j] * 0.94)
            for i in range(self.row):
                if img[i, j] >= rgb:
                    img[i, j] = 255
        # return processed image
        return img

    def _remove_strikethrough(self, img):
        """
        Removes strikethrough line from image. RGB value of the line is 
        mostly '0', RGB threshold value of '10' is taken.
        """
        # get black color pixel coordinates
        st_line = np.column_stack(np.where(img < 10))
        # add neigbour pixels
        nghb = []
        for i, j in st_line:
            nghb.append([i, max(j-1, 0)])
            nghb.append([i, min(j+1, self.col-1)])
            nghb.append([max(i-1, 0), j])
            nghb.append([min(i+1, self.row-1), j])
        # append neighbour pixels to black color pixels
        st_line = np.unique(np.append(st_line, nghb, axis=0), axis=0)
        # remove black pixels and their neighbours from image
        for i, j in st_line:
            img[i, j] = 255
        # return processed image
        return img

    def _text_borders(self, bpx):
        """
        Finds left and right borders of the captcha text. 
        """
        # left border of the captcha text
        left = count = 0
        for i, nr in enumerate(bpx):
            if nr > 0:
                if not count:
                    left = i
                    count += 1
                elif count < 5:
                    count += 1
                else:
                    break
            else:
                left = count = 0

        # right border of the captcha text
        right = self.col - 1
        count = 0
        for i, nr in reversed(list(enumerate(bpx))):
            if nr > 0:
                if not count:
                    right = i
                    count += 1
                elif count < 5:
                    count += 1
                else:
                    break
            else:
                right = count = 0
        # return boundaries
        return (left, right)

    def _fix_char_borders(self, chars):
        """
        Fixes the borders of collided characters.
        """
        # get chars that have width greater than the threshold
        lch = list(
            filter(lambda tpl: (tpl[1]-tpl[0] >= self.width_thr), chars))

        # If captcha is splitted into 5 chars, return char borders.
        # Else, check distance between borders and split to chars.
        # Add defined bandwith to splitted borders to avoid overlap losses.
        bw = 3
        if len(chars) == 5 and not lch:
            pass
        elif len(chars) == 4 and len(lch) == 1:
            i = chars.index(lch[0])
            b1, b2 = lch[0]
            sp = round((b2 - b1)/2)
            chars[i+1:i+1] = [(b1, b1+sp+bw), (b1+sp-bw, b2)]
            del chars[i]
        elif len(chars) == 3 and len(lch) == 1:
            i = chars.index(lch[0])
            b1, b2 = lch[0]
            sp = round((b2 - b1)/3)
            chars[i+1:i+1] = [(b1, b1+sp+bw), (b1+sp-bw, b1+2*sp+bw),
                              (b1+2*sp-bw, b2)]
            del chars[i]
        elif len(chars) == 3 and len(lch) == 2:
            for tpl in lch:
                i = chars.index(tpl)
                b1, b2 = tpl
                sp = round((b2 - b1)/2)
                chars[i+1:i+1] = [(b1, b1+sp+bw), (b1+sp-bw, b2)]
                del chars[i]
        elif len(chars) == 2 and len(lch) == 1:
            i = chars.index(lch[0])
            b1, b2 = lch[0]
            sp = round((b2 - b1)/4)
            chars[i+1:i+1] = [(b1, b1+sp+bw), (b1+sp-bw, b1+2*sp+bw),
                              (b1+2*sp-bw, b1+3*sp+bw), (b1+3*sp-bw, b2)]
            del chars[i]
        elif len(chars) == 2 and len(lch) == 2:
            # find the wider one which contains 3 chars
            lch.sort(key=lambda t: t[1] - t[0], reverse=True)
            # split wider one into 3 chars
            i = chars.index(lch[0])
            b1, b2 = lch[0]
            sp = round((b2 - b1)/3)
            chars[i+1:i+1] = [(b1, b1+sp+bw), (b1+sp-bw, b1+2*sp+bw),
                              (b1+2*sp-bw, b2)]
            del chars[i]
            # split the other into two chars
            i = chars.index(lch[1])
            b1, b2 = lch[0]
            sp = round((b2 - b1)/2)
            chars[i+1:i+1] = [(b1, b1+sp+bw), (b1+sp-bw, b2)]
            del chars[i]
        elif len(chars) == 1 and len(lch) == 1:
            b1, b2 = lch[0]
            sp = round((b2 - b1)/5)
            chars = [(b1, b1+sp+bw), (b1+sp-bw, b1+2*sp+bw),
                     (b1+2*sp-bw, b1+3*sp+bw), (b1+3*sp-bw, b1+4*sp+bw),
                     (b1+4*sp-bw, b2)]
        else:
            return False
        # return modified borders
        return chars

    def _char_borders(self, img):
        """
        Finds borders for each character.
        """
        # count number of black pixels in each column
        bpx = np.count_nonzero(img == 0, axis=0)

        # get captcha text left and right borders
        left, right = self._text_borders(bpx)

        # get white space column indexes
        wpx = np.where(bpx == 0)[0]
        wpx = wpx[(wpx > left) & (wpx < right)]

        # find border indexes
        borders = []
        b1 = b2 = wpx[0]
        for v in wpx:
            if v - b2 > 1:
                borders.append((b1, b2))
                b1 = b2 = v
            else:
                b2 = v
        borders.append((b1, b2))

        # split to chars
        chars = []
        c1 = left
        for b1, b2 in borders:
            chars.append((c1, b1-1))
            c1 = b2 + 1
        chars.append((c1, right))

        # get char widths
        wch = [tpl[1]-tpl[0] for tpl in chars]
        lch = []
        for i, w in enumerate(wch):
            # if there are narrow char borders, merge it with its neighbour
            if w < 5:
                b1, b2 = chars[i]
                # sum of the black pixels within borders
                sum = 0
                for k in range(b1, b2+1):
                    sum += bpx[k]
                if i == 0:
                    # if it is the first char, merge it with the next char
                    nb1, nb2 = chars[1]
                    chars[1] = (b1, nb2)
                elif i == len(chars) - 1:
                    # if it is the last char, merge it with the previous char
                    pb1, pb2 = chars[-2]
                    chars[-2] = (pb1, b2)
                elif sum > 5:
                    # ignore small ones
                    # merge with the neighbour which has higher number of
                    # black pixels in its neighbour border
                    pb1, pb2 = chars[i-1]
                    nb1, nb2 = chars[i+1]
                    if bpx[pb2] > bpx[nb1]:
                        chars[i-1] = (pb1, b2)
                    else:
                        chars[i+1] = (b1, nb2)
                del chars[i]

        # return char borders
        return self._fix_char_borders(chars)

    def _split_to_chars(self, img):
        """
        Split captcha text into characters by finding character borders. 
        """
        # get character's left and right borders
        chars = self._char_borders(img)

        if not chars:
            raise Exception

        for i, char in enumerate(chars):
            b1, b2 = char
            # trim chars wider than the threshold
            tr = b2 - b1 + 1 - self.width_thr
            if tr > 0:
                b1 += int(tr/2) + tr % 2
                b2 -= int(tr/2)
            # count number of black pixels in each row for each character
            bpx = np.count_nonzero(img[:, b1:b2+1] == 0, axis=1)
            # top border of the character
            top = 0
            for r, nr in enumerate(bpx):
                if nr > 1:
                    top = r
                    break
            # bottom border of the character
            bottom = self.row - 1
            for r, nr in reversed(list(enumerate(bpx))):
                if nr > 1:
                    bottom = r
                    break
            # set new borders
            chars[i] = ((b1, top), (b2, bottom))
        # return char border coordinates as ((x1,y1),(x2,y2))
        return chars

    def _crop_image(self, img, coord):
        """
        (0,0) coordinate is the top-left corner of image.
        coord = ((x1,y1),(x2,y2))
        """
        x1, y1 = coord[0]
        x2, y2 = coord[1]
        return img[y1:min(y2+1, self.row), x1:min(x2+1, self.col)]

    def process(self, filepath):
        # load the image and convert it to grayscale
        img = cv2.imread(filepath)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # get size of image
        self.row, self.col = img.shape

        # remove background color
        img = self._remove_background(img)
        # remove strikethrough line
        img = self._remove_strikethrough(img)

        # convert image to black-white
        mask = img < 255
        img[mask] = 0

        try:
            chars = self._split_to_chars(img)
        except:
            return False

        characters = []
        for char in chars:
            char_img = self._crop_image(img, char)
            # add white borders to get same size character images
            h, w = char_img.shape
            top = self.row - h
            right = self.width_thr - w
            char_img = cv2.copyMakeBorder(
                char_img, top, 0, 0, right, cv2.BORDER_CONSTANT, None, 255)
            characters.append(char_img)
        # return cropped character images
        return characters
