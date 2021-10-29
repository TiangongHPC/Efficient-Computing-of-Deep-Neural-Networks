import argparse
import json
import time
import os
import glob
import onnx
import onnxruntime

import numpy as np
import cv2

CHARS = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 
        'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 
        'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 'A', 'B', 
        'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 
        'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
COLOR_MAP = {CHARS[idx] : (np.random.random(3) * 255).astype(np.uint8).tolist() for idx in range(len(CHARS))}

def calculate_iou(box1, box2):
    if len(box1) == 8:
        box1 = [[box1[2 * i], box1[2 * i + 1]] for i in range(4)]
    if len(box2) == 8:
        box2 = [[box2[2 * i], box2[2 * i + 1]] for i in range(4)]

    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for i in range(4):
        xmin = min(xmin, box1[i][0], box2[i][0])
        ymin = min(ymin, box1[i][1], box2[i][1])
        xmax = max(xmax, box1[i][0], box2[i][0])
        ymax = max(ymax, box1[i][1], box2[i][1])
    w, h = int(xmax-xmin+1), int(ymax-ymin+1)

    pic1 = np.zeros((h, w), dtype=np.uint8)
    pic2 = np.zeros_like(pic1)
    b1 = [[a[0] - xmin, a[1] - ymin] for a in box1]
    b2 = [[a[0] - xmin, a[1] - ymin] for a in box2]

    cv2.fillPoly(pic1,[np.array(b1,np.int32)],1)
    cv2.fillPoly(pic2,[np.array(b2,np.int32)],1)

    inter = cv2.bitwise_and(pic1, pic2).sum()
    if inter == 0:
        return 0
    union = cv2.bitwise_or(pic1, pic2).sum()
    return inter / union

def nms(results, threshold):
    if len(results) == 0:
        return []
    bboxs = [res[0] for res in results]
    scores = [res[1] for res in results]
    bboxs = np.array(bboxs)
    scores = np.array(scores)

    keep = [True] * len(scores)
    for i in range(len(scores)):
        if keep[i] == False:
            continue
        for j in range(i+1, len(scores)):
            iou = calculate_iou(bboxs[i], bboxs[j])
            if iou > threshold:
                keep[j] = False
    return keep

def img_preprocess(image, min_size=800, max_size=1333):
    size = min_size
    h, w, _ = image.shape
    scale = size * 1.0 / min(h, w)
    if h < w:
        newh, neww = size, scale * w
    else:
        newh, neww = scale * h, size
    if max(newh, neww) > max_size:
        scale = max_size * 1.0 / max(newh, neww)
        newh = newh * scale
        neww = neww * scale
    neww = int(neww + 0.5)
    newh = int(newh + 0.5)
    out = cv2.resize(image, (neww, newh), interpolation=cv2.INTER_LINEAR_EXACT)
    return np.array(out, np.float32).transpose((2,0,1))

def points_euclidean_distance(p1, p2):
    '''
    calculate distance of two points
    '''
    dist = np.linalg.norm(np.array(p1) - np.array(p2))
    return dist

def line_intersect_rect(pa, pb, box):
    '''
    求线段和四边形的交点并返回,若无交点返回None
    '''
    box = np.array(box).reshape(4, 2)
    pa, pb = np.array(pa), np.array(pb)
    for i in range(4):
        pc, pd = box[i], box[(i+1)%4]
        dpa = pb-pa
        dpd = pc-pd
        delta = np.cross(dpa, dpd) # (xb-xa)*(yc-yd) - (xc-xd)*(yb-ya)

        if np.abs(delta) < 0.0001:  # delta = 0, paralell
            continue
        delta = 1.0/delta
        dp = pc-pa

        alpha = delta * np.cross(dp, dpd)
        beta = delta * np.cross(dpa, dp)

        crossP = pa + alpha*dpa # intersect
        if alpha >= 0 and alpha <= 1 and beta >= 0 and beta <= 1:
            return crossP
    return None

class CharPoly:
    def __init__(self, char, score, poly):
        self.char = char
        self.score = score
        self.poly = np.array(poly)
        self.center = np.mean(self.poly, axis=0)
        self.minx = min(self.poly[:, 0])
        self.maxx = max(self.poly[:, 0])
        self.miny = min(self.poly[:, 1])
        self.maxy = max(self.poly[:, 1])
        self.max_edge = 0
        for i, p in enumerate(self.poly):
            self.max_edge = max(points_euclidean_distance(p, self.poly[(i+1)%len(self.poly)]), self.max_edge)
    def distance(self, other):
        return np.sqrt(np.sum(np.power(self.center - other.center, 2)))
    def interior_distance(self, other):
        p1, p2 = self.center, other.center
        b1, b2 = self.poly, other.poly
        dist = points_euclidean_distance(p1, p2)

        p3 = line_intersect_rect(p1, p2, b1)
        p4 = line_intersect_rect(p1, p2, b2)
        if p3 is not None and p4 is not None:
            dist = dist - points_euclidean_distance(p1, p3) - points_euclidean_distance(p2, p4)
        else:
            dist = 0
        return dist
    def __str__(self):
        return f"{self.char} {self.poly} {self.center}"

class UF:
    """
    并查集, 基于size优化
    """
    def __init__(self, n):
        self.fa = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        if self.fa[x] != x:
            self.fa[x] = self.find(self.fa[x])
        return self.fa[x]

    def connected(self, x, y):
        return self.find(x) == self.find(y)

    def unite(self, x, y):
        fx = self.find(x)
        fy = self.find(y)
        if fx == fy:
            return
        if self.sz[fx] > self.sz[fy]:
            self.fa[fy] = fx
            self.sz[fx] += self.sz[fy]
        else:
            self.fa[fx] = fy
            self.sz[fy] += self.sz[fx]

    def get_cc(self):
        cc = {}
        for i, x in enumerate(self.fa):
            fx = self.fa[x]
            if fx not in cc:
                cc[fx] = [i]
            else:
                cc[fx].append(i)
        return list(cc.values())

def parseOutput(predictions, min_score, height, width, h, w, use_nms=False, nms_iou=0.7):
    scale_x, scale_y = (width / w, height / h)
    preds = []
    for i, (box, cls_idx, score) in enumerate(zip(predictions[0], predictions[1], predictions[2])):
        score = float(score)
        if score < min_score:
            continue
        x1, y1, x2, y2 = box
        x1 = max(0, min(int(x1 * scale_x), width - 1))
        x2 = max(0, min(int(x2 * scale_x), width - 1))
        y1 = max(0, min(int(y1 * scale_y), height - 1))
        y2 = max(0, min(int(y2 * scale_y), height - 1))
        bbox = np.array([[x1,y1], [x2, y1], [x2, y2], [x1, y2]])
        preds.append([bbox, score, cls_idx])
    result = []
    if use_nms:
        keep = nms(preds, nms_iou)
        for k, res in zip(keep, preds):
            if k == True:
                result.append(CharPoly(CHARS[res[2]], res[1], res[0]))
    else:
        for res in preds:
            result.append(CharPoly(CHARS[res[2]], res[1], res[0]))

    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='./test_imgs/5.png')
    parser.add_argument('--onnx-file', '-onnx', default='optFcos.onnx')
    parser.add_argument('--score_th', type=float, default=0.4)
    parser.add_argument('--nms_iou', type=float, default=0.5)
    parser.add_argument('--save_dir', type=str, default='tmp')
    parser.add_argument('--save_img',action="store_true", default=False)
    
    args = parser.parse_args()
    
    assert os.path.exists(args.data)
    assert os.path.exists(args.onnx_file)

    if args.save_img:
        os.makedirs(args.save_dir, exist_ok=True)

    ############# load model ################
    model = onnx.load(args.onnx_file)
    onnx.checker.check_model(model)
    ort_session = onnxruntime.InferenceSession(args.onnx_file)
    print(f"load model {args.onnx_file}")

    img_paths = []
    if os.path.isfile(args.data):
        img_paths = [args.data]
    elif os.path.isdir(args.data):
        img_paths += glob.glob(os.path.join(args.data, "*.jpg"))
        img_paths += glob.glob(os.path.join(args.data, "*.png"))
        img_paths += glob.glob(os.path.join(args.data, "*.bmp"))
    img_paths.sort()
    assert len(img_paths) > 0

    results = []
    for img_idx, img_path in enumerate(img_paths):
        filename = os.path.basename(img_path)
        t1 = time.time()
        ori_image = cv2.imread(img_path)
        image = ori_image
        res_img = image.copy()
        height, width, _ = image.shape
        img_np = img_preprocess(image, min_size=800, max_size=1333)
        _, h, w = img_np.shape
        
        ort_inputs = {ort_session.get_inputs()[0].name: img_np}
        ort_outs = ort_session.run(None, ort_inputs)
        ori_res = parseOutput(ort_outs, args.score_th, height, width, h, w, use_nms=True, nms_iou=args.nms_iou) 


        uf = UF(len(ori_res))
        for i in range(len(ori_res)):
            for j in range(i+1, len(ori_res)):
                dist = ori_res[i].interior_distance(ori_res[j])
                if ori_res[i].maxy < ori_res[j].miny or ori_res[i].miny > ori_res[j].maxy:
                    continue
                if dist < ori_res[i].max_edge * 2:
                    uf.unite(i, j)
 
        uf_res = uf.get_cc()

        det_lines = []
        for _, ele in enumerate(uf_res):
            if len(ele) < 3:
                continue
            line_char_polys = []
            for x in ele:
                line_char_polys.append(ori_res[x])
            line_char_polys.sort(key=lambda x: x.center[0])

            det_lines.append(line_char_polys)

        for a in det_lines:
            for b in a:
                print(b.char, end="")
            print("")

        det_lines.sort(key=lambda x: x[0].center[1])

        print()

        for a in det_lines:
            for b in a:
                print(b.char, end="")
            print("")

        post_res = []
        pre_line = []
        for line_char_polys in det_lines:
            if len(pre_line) > 0:
                min_dis = h
                for char_poly in line_char_polys:
                    for pre_char in pre_line:
                        dis = char_poly.interior_distance(pre_char)
                        min_dis = min(min_dis, dis)
                if min_dis > pre_line[0].max_edge * 2:
                    continue
            pre_line = line_char_polys
            post_res.extend(line_char_polys)

        chars = []
        for res in post_res:
            chars.append(res.char)

        print(f"{img_idx}, time={time.time()-t1}s, {img_path}, {''.join(chars)}")

        if args.save_img:
            white_img = np.ones_like(res_img) * 255
            for res in post_res:
                cv2.polylines(res_img, [res.poly.astype(np.int32)], True, (0, 0, 255), 1)
                cv2.putText(white_img, res.char, res.center.astype('int32').tolist(),
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=1,
                            color=(0,0,255),
                            thickness=2)
            res_img = np.concatenate([res_img, np.array(white_img)], axis=1)
            save_img_path = os.path.join(args.save_dir, filename) 
            cv2.imwrite(save_img_path, res_img)   
    
    