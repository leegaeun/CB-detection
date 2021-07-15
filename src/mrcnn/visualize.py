"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import random
import itertools
import colorsys
import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from IPython import get_ipython
ipy = get_ipython()
if ipy is not None:
    ipy.run_line_magic('matplotlib', 'inline')
#plt.switch_backend('agg')
import matplotlib.patches as patches
import matplotlib.lines as lines
from matplotlib.patches import Polygon
import IPython.display

import skimage.color
from skimage.morphology import skeletonize, label
import cv2

from mrcnn import utils
import Utils_custom as us

import keras.backend as K
import keras.callbacks as KC
from IPython.display import clear_output

############################################################
#  Plotting _ 
############################################################
class Plotting_custom(KC.Callback):

    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.rpn_class_loss = []
        self.val_rpn_class_loss = []
        self.rpn_bbox_loss = []
        self.val_rpn_bbox_loss = []
        self.mrcnn_class_loss = []
        self.val_mrcnn_class_loss = []
        self.mrcnn_bbox_loss = []
        self.val_mrcnn_bbox_loss = []
        self.mrcnn_mask_loss = []
        self.val_mrcnn_mask_loss = []
        self.lr = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.rpn_class_loss.append(logs.get('rpn_class_loss'))
        self.val_rpn_class_loss.append(logs.get('val_rpn_class_loss'))
        self.rpn_bbox_loss.append(logs.get('rpn_bbox_loss'))
        self.val_rpn_bbox_loss.append(logs.get('val_rpn_bbox_loss'))
        self.mrcnn_class_loss.append(logs.get('mrcnn_class_loss'))
        self.val_mrcnn_class_loss.append(logs.get('val_mrcnn_class_loss'))
        self.mrcnn_bbox_loss.append(logs.get('mrcnn_bbox_loss'))
        self.val_mrcnn_bbox_loss.append(logs.get('val_mrcnn_bbox_loss'))
        self.mrcnn_mask_loss.append(logs.get('mrcnn_mask_loss'))
        self.val_mrcnn_mask_loss.append(logs.get('val_mrcnn_mask_loss'))
        self.lr.append(K.get_value(self.model.optimizer.lr))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.rpn_class_loss, label="rpn_class_loss")
        plt.plot(self.x, self.val_rpn_class_loss, label="val_rpn_class_loss")
        plt.plot(self.x, self.rpn_bbox_loss, label="rpn_bbox_loss")
        plt.plot(self.x, self.val_rpn_bbox_loss, label="val_rpn_bbox_loss")
        plt.plot(self.x, self.mrcnn_class_loss, label="mrcnn_class_loss")
        plt.plot(self.x, self.val_mrcnn_class_loss, label="val_mrcnn_class_loss")
        plt.plot(self.x, self.mrcnn_bbox_loss, label="mrcnn_bbox_loss")
        plt.plot(self.x, self.val_mrcnn_bbox_loss, label="val_mrcnn_bbox_loss")
        plt.plot(self.x, self.mrcnn_mask_loss, label="mrcnn_mask_loss")
        plt.plot(self.x, self.val_mrcnn_mask_loss, label="val_mrcnn_mask_loss")
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(ncol=1, loc='center left',bbox_to_anchor=(1,1))
        plt.show();
        plt.plot(self.x, self.lr, label="learning_rate")
        plt.title('Learning rate')
        plt.ylabel('learning rate')
        plt.xlabel('epoch')
        plt.grid(True)
        plt.legend(loc=0)
        plt.show();                        
                
    #def on_train_end(self, logs={}):
    #    fig.savefig('plot.pdf')    

############################################################
#  Visualization
############################################################

def display_images(images, titles=None, cols=4, cmap=None, norm=None,
                   interpolation=None):
    """Display the given set of images, optionally with titles.
    images: list or array of image tensors in HWC format.
    titles: optional. A list of titles to display with each image.
    cols: number of images per row
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interporlation to use for display.
    """
    titles = titles if titles is not None else [""] * len(images)
    rows = len(images) // cols + 1
    plt.figure(figsize=(14, 14 * rows // cols))
    i = 1
    for image, title in zip(images, titles):
        plt.subplot(rows, cols, i)
        plt.title(title, fontsize=9)
        plt.axis('off')
        plt.imshow(image.astype(np.uint8), cmap=cmap,
                   norm=norm, interpolation=interpolation)
        i += 1
    plt.show()


def random_colors(N, bright=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    #random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha):
    """Apply the given mask to the image.
    """
    # 'c' is one of R,G,and B channel.
    for c in range(image.shape[2]):
        image[:, :, c] = np.where(mask == 1,
                                  image[:, :, c] * (1 - alpha) + alpha * color[c] * 255,
                                  image[:, :, c])
    return image



def save_result_figures(image, result, class_names, save_filename,
                        truemasks = None, truemasks_class_id = None,
                        captions=None, show_mask=True, show_bbox=True, title="", figsize=(200, 200), colors=None,  
                        figDir = None, fig_option='all'):
    plt.switch_backend('agg')
    """
    boxes in result: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks in result: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    truemasks: (optional) [height, width, num_instances] ground-truth mask in image
    truemasks_class_id: [num_instances] in image
    captions: (optional) A list of strings to use as captions for each object
    overlalyDir: (optional) directory of image + instance-masks (just overlay)
    figDir: (optional) directory of image + instance-masks + bboxs + captions, ... with colors
    fig_option: 'all', 'gt', 'ai'
    """
    
    _, ax = plt.subplots(1, 1, figsize=figsize)
    #plt.subplots_adjust(left = 0, bottom = 0, right = 1, top = 1, hspace = 0, wspace = 0)
    
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 0, -0)
    ax.set_xlim(-0, width + 0)
    ax.axis('off')
    ax.set_title(title)
    
    fig.add_axes(ax)
    
    masked_image = image.astype(np.uint32).copy()
    masked_image = skimage.color.gray2rgb(masked_image)
    
    
    ########################### 
    ###### Ground-truth #######
    ########################### 
    if fig_option=='ai':
        truemasks = None
    if truemasks is not None: 
        #color = colors[N] # End of random color : Ground-truth
        color = (1.0, 1.0, 1.0)
        
        for i in range(truemasks.shape[2]):
            class_id = truemasks_class_id[i]
            label = class_names[class_id]
            
            _, truemask = cv2.threshold(truemasks[:, :, i], 0, 1, cv2.THRESH_BINARY)
            
            '''Bounding box'''
            bbox_gt = utils.extract_bboxes(np.expand_dims(truemask,-1))[0]
            y1 = bbox_gt[0]; x1 = bbox_gt[1]; y2 = bbox_gt[2]; x2 = bbox_gt[3];
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=0.5,
                                      alpha=0.7, linestyle="dashed",
                                      edgecolor=color, facecolor='none')
                ax.add_patch(p)
            
            '''Caption'''
            if not captions:
                caption = "{}".format(label)
            else:
                caption = caption[i]
            ax.text(x1, y1 -5, caption, color=color, size=9, backgroundcolor='none')
            
            '''Mask & Polygon'''
            if not show_mask:
                continue
            padded_truemask = np.zeros((truemask.shape[0] + 2, truemask.shape[1] + 2), dtype=np.uint8)
            padded_truemask[1:-1, 1:-1] = truemask
            contours = find_contours(padded_truemask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, linewidth = 0.5, facecolor='none', edgecolor=color,alpha = 1)
                ax.add_patch(p)
#                 p = Polygon(verts, linewidth = 0.5, facecolor=color, edgecolor='none',alpha = 0.5)
#                 ax.add_patch(p)
    
    
    ########################### 
    ####### Prediction ########
    ########################### 
    
    boxes = np.array(result['rois'])
    masks = np.array(result['masks'])
    class_ids = result['class_ids']
    scores = result['scores']
    

    # Number of instances
    N = boxes.shape[0]
#     if not N:
#         print("\n*** No instances to display *** \n")
#         return 
#     else:
#         #print(boxes.shape[0], masks.shape[-1], class_ids.shape[0])
#         assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]
    
    '''Generate random colors'''
    colors = colors or random_colors(N)
    
    if any(fig_option==np.asarray(['all','ai'])):
        for i in range(N):
            color = colors[i] # i'th of random color : prediction(i) #GAENU
            box = boxes[i]
            mask = masks[:, :, i]
            class_id = class_ids[i]
            #score = scores[i]
            

            '''Bounding box'''
            if not np.any(box):
                # Skip this instance. Has no bbox. Likely lost in image cropping.
                continue
            y1, x1, y2, x2 = box
            if show_bbox:
                p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=0.5,
                                    alpha=0.7, linestyle="dashed",
                                    edgecolor=color, facecolor='none')
                ax.add_patch(p)

            '''Caption'''
            if not captions:
                label = class_names[class_id]
#                 score = scores[i] if scores is not None else None
#                 caption = "{} {:.3f}".format(label, score) if score else label
                caption = "{}".format(label)
            else:
                caption = captions[i]
            ax.text(x1, y1-5, caption, color=color, size=9, backgroundcolor='none')

#             '''Mask_skeletonized'''
#             _, mask = cv2.threshold(mask, 0, 1, cv2.THRESH_BINARY)
#             mask_skl = np.copy(mask)
#             mask_skl = skeletonize(mask_skl)
#             masked_image = apply_mask(masked_image, mask_skl, color, alpha=1)

            '''Mask & Polygon'''
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros((mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, linewidth = 0.5, facecolor='none', edgecolor=color,alpha = 1)
                ax.add_patch(p)
                p = Polygon(verts, linewidth = 0.5, facecolor=color, edgecolor='none',alpha = 0.5)
                ax.add_patch(p)
        
#     ax.imshow(masked_image.astype(np.uint8)[:,:,0], cmap = 'gray')
    
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.imshow(masked_image.astype(np.uint8))
    
    '''Save'''
#     masked_image = np.asarray(masked_image, dtype = "uint8")
#     if(overlayDir is not None) :
#         OverlayFile = overlayDir + "/" + save_filename
#         cv2.imwrite(OverlayFile, masked_image)
    
    if figDir is not None:
        plt.savefig(os.path.join(figDir,save_filename), dpi = 300, bbox_inches='tight', pad_inches=0)
        #plt.close(fig)
        plt.cla()

        

        
def save_result_masks(result, class_names, save_filename, maskDir, crop_meta=None):
    """
    maskDir: Directory of predicted instance mask.
    crop_meta: []
    """
    masks = np.array(result['masks'],dtype='uint8')
    class_ids = result['class_ids']
    
    # Save (Mask : each class)
    masks = us.restore_image_withCropmeta(masks, crop_meta=crop_meta)
    
    for i, class_id in enumerate (class_ids):
        mask = masks[:,:,i]
        class_name = class_names[class_id]
    
        maskDir_clss = os.path.join(maskDir, class_name)
        if not os.path.isdir(maskDir_clss):
            os.mkdir(maskDir_clss)
    
        _, mask = cv2.threshold(np.asarray(mask,dtype='uint8'), 0, 1, cv2.THRESH_BINARY)
        cv2.imwrite(os.path.join(maskDir_clss,save_filename), mask)
        
        
    

def draw_rois(image, rois, refined_rois, mask, class_ids, class_names, limit=10):
    """
    anchors: [n, (y1, x1, y2, x2)] list of anchors in image coordinates.
    proposals: [n, 4] the same anchors but refined to fit objects better.
    """
    masked_image = image.copy()

    # Pick random anchors in case there are too many.
    ids = np.arange(rois.shape[0], dtype=np.int32)
    ids = np.random.choice(
        ids, limit, replace=False) if ids.shape[0] > limit else ids

    fig, ax = plt.subplots(1, figsize=(12, 12))
    if rois.shape[0] > limit:
        plt.title("Showing {} random ROIs out of {}".format(
            len(ids), rois.shape[0]))
    else:
        plt.title("{} ROIs".format(len(ids)))

    # Show area outside image boundaries.
    ax.set_ylim(image.shape[0] + 20, -20)
    ax.set_xlim(-50, image.shape[1] + 20)
    ax.axis('off')

    for i, id in enumerate(ids):
        color = np.random.rand(3)
        class_id = class_ids[id]
        # ROI
        y1, x1, y2, x2 = rois[id]
        p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                              edgecolor=color if class_id else "gray",
                              facecolor='none', linestyle="dashed")
        ax.add_patch(p)
        # Refined ROI
        if class_id:
            ry1, rx1, ry2, rx2 = refined_rois[id]
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal for easy visualization
            ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

            # Label
            label = class_names[class_id]
            ax.text(rx1, ry1 + 8, "{}".format(label),
                    color='w', size=11, backgroundcolor="none")

            # Mask
            m = utils.unmold_mask(mask[id], rois[id]
                                  [:4].astype(np.int32), image.shape)
            masked_image = apply_mask(masked_image, m, color)

    ax.imshow(masked_image)

    # Print stats
    print("Positive ROIs: ", class_ids[class_ids > 0].shape[0])
    print("Negative ROIs: ", class_ids[class_ids == 0].shape[0])
    print("Positive Ratio: {:.2f}".format(
        class_ids[class_ids > 0].shape[0] / class_ids.shape[0]))


# TODO: Replace with matplotlib equivalent?
def draw_box(image, box, color):
    """Draw 3-pixel width bounding boxes on the given image array.
    color: list of 3 int values for RGB.
    """
    y1, x1, y2, x2 = box
    image[y1:y1 + 2, x1:x2] = color
    image[y2:y2 + 2, x1:x2] = color
    image[y1:y2, x1:x1 + 2] = color
    image[y1:y2, x2:x2 + 2] = color
    return image


def display_top_masks(image, mask, class_ids, class_names, limit=4):
    """Display the given image and the top few class masks."""
    to_display = []
    titles = []
    to_display.append(image)
    titles.append("H x W={}x{}".format(image.shape[0], image.shape[1]))
    # Pick top prominent classes in this image
    unique_class_ids = np.unique(class_ids)
    mask_area = [np.sum(mask[:, :, np.where(class_ids == i)[0]])
                 for i in unique_class_ids]
    top_ids = [v[0] for v in sorted(zip(unique_class_ids, mask_area),
                                    key=lambda r: r[1], reverse=True) if v[1] > 0]
    # Generate images and titles
    for i in range(limit):
        class_id = top_ids[i] if i < len(top_ids) else -1
        # Pull masks of instances belonging to the same class.
        m = mask[:, :, np.where(class_ids == class_id)[0]]
        m = np.sum(m * np.arange(1, m.shape[-1] + 1), -1)
        to_display.append(m)
        titles.append(class_names[class_id] if class_id != -1 else "-")
    display_images(to_display, titles=titles, cols=limit + 1, cmap="Blues_r")


def plot_precision_recall(AP, precisions, recalls):
    """Draw the precision-recall curve.

    AP: Average precision at IoU >= 0.5
    precisions: list of precision values
    recalls: list of recall values
    """
    # Plot the Precision-Recall curve
    _, ax = plt.subplots(1)
    ax.set_title("Precision-Recall Curve. AP@50 = {:.3f}".format(AP))
    ax.set_ylim(0, 1.1)
    ax.set_xlim(0, 1.1)
    _ = ax.plot(recalls, precisions)


def plot_overlaps(gt_class_ids, pred_class_ids, pred_scores,
                  overlaps, class_names, threshold=0.5):
    """Draw a grid showing how ground truth objects are classified.
    gt_class_ids: [N] int. Ground truth class IDs
    pred_class_id: [N] int. Predicted class IDs
    pred_scores: [N] float. The probability scores of predicted classes
    overlaps: [pred_boxes, gt_boxes] IoU overlaps of predictins and GT boxes.
    class_names: list of all class names in the dataset
    threshold: Float. The prediction probability required to predict a class
    """
    gt_class_ids = gt_class_ids[gt_class_ids != 0]
    pred_class_ids = pred_class_ids[pred_class_ids != 0]

    plt.figure(figsize=(12, 10))
    plt.imshow(overlaps, interpolation='nearest', cmap=plt.cm.Blues)
    plt.yticks(np.arange(len(pred_class_ids)),
               ["{} ({:.2f})".format(class_names[int(id)], pred_scores[i])
                for i, id in enumerate(pred_class_ids)])
    plt.xticks(np.arange(len(gt_class_ids)),
               [class_names[int(id)] for id in gt_class_ids], rotation=90)

    thresh = overlaps.max() / 2.
    for i, j in itertools.product(range(overlaps.shape[0]),
                                  range(overlaps.shape[1])):
        text = ""
        if overlaps[i, j] > threshold:
            text = "match" if gt_class_ids[j] == pred_class_ids[i] else "wrong"
        color = ("white" if overlaps[i, j] > thresh
                 else "black" if overlaps[i, j] > 0
                 else "grey")
        plt.text(j, i, "{:.3f}\n{}".format(overlaps[i, j], text),
                 horizontalalignment="center", verticalalignment="center",
                 fontsize=9, color=color)

    plt.tight_layout()
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")


def draw_boxes(image, boxes=None, refined_boxes=None,
               masks=None, captions=None, visibilities=None,
               title="", ax=None):
    """Draw bounding boxes and segmentation masks with differnt
    customizations.

    boxes: [N, (y1, x1, y2, x2, class_id)] in image coordinates.
    refined_boxes: Like boxes, but draw with solid lines to show
        that they're the result of refining 'boxes'.
    masks: [N, height, width]
    captions: List of N titles to display on each box
    visibilities: (optional) List of values of 0, 1, or 2. Determine how
        prominant each bounding box should be.
    title: An optional title to show over the image
    ax: (optional) Matplotlib axis to draw on.
    """
    # Number of boxes
    assert boxes is not None or refined_boxes is not None
    N = boxes.shape[0] if boxes is not None else refined_boxes.shape[0]

    # Matplotlib Axis
    if not ax:
        _, ax = plt.subplots(1, figsize=(12, 12))

    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    margin = image.shape[0] // 10
    ax.set_ylim(image.shape[0] + margin, -margin)
    ax.set_xlim(-margin, image.shape[1] + margin)
    ax.axis('off')

    ax.set_title(title)

    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        # Box visibility
        visibility = visibilities[i] if visibilities is not None else 1
        if visibility == 0:
            color = "gray"
            style = "dotted"
            alpha = 0.5
        elif visibility == 1:
            color = colors[i]
            style = "dotted"
            alpha = 1
        elif visibility == 2:
            color = colors[i]
            style = "solid"
            alpha = 1

        # Boxes
        if boxes is not None:
            if not np.any(boxes[i]):
                # Skip this instance. Has no bbox. Likely lost in cropping.
                continue
            y1, x1, y2, x2 = boxes[i]
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=alpha, linestyle=style,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Refined boxes
        if refined_boxes is not None and visibility > 0:
            ry1, rx1, ry2, rx2 = refined_boxes[i].astype(np.int32)
            p = patches.Rectangle((rx1, ry1), rx2 - rx1, ry2 - ry1, linewidth=2,
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)
            # Connect the top-left corners of the anchor and proposal
            if boxes is not None:
                ax.add_line(lines.Line2D([x1, rx1], [y1, ry1], color=color))

        # Captions
        if captions is not None:
            caption = captions[i]
            # If there are refined boxes, display captions on them
            if refined_boxes is not None:
                y1, x1, y2, x2 = ry1, rx1, ry2, rx2
            x = random.randint(x1, (x1 + x2) // 2)
            ax.text(x1, y1, caption, size=11, verticalalignment='top',
                    color='w', backgroundcolor="none",
                    bbox={'facecolor': color, 'alpha': 0.5,
                          'pad': 2, 'edgecolor': 'none'})

        # Masks
        if masks is not None:
            mask = masks[:, :, i]
            masked_image = apply_mask(masked_image, mask, color)
            # Mask Polygon
            # Pad to ensure proper polygons for masks that touch image edges.
            padded_mask = np.zeros(
                (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
            padded_mask[1:-1, 1:-1] = mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8))


def display_table(table):
    """Display values in a table format.
    table: an iterable of rows, and each row is an iterable of values.
    """
    html = ""
    for row in table:
        row_html = ""
        for col in row:
            row_html += "<td>{:40}</td>".format(str(col))
        html += "<tr>" + row_html + "</tr>"
    html = "<table>" + html + "</table>"
    IPython.display.display(IPython.display.HTML(html))


def display_weight_stats(model):
    """Scans all the weights in the model and returns a list of tuples
    that contain stats about each weight.
    """
    layers = model.get_trainable_layers()
    table = [["WEIGHT NAME", "SHAPE", "MIN", "MAX", "STD"]]
    for l in layers:
        weight_values = l.get_weights()  # list of Numpy arrays
        weight_tensors = l.weights  # list of TF tensors
        for i, w in enumerate(weight_values):
            weight_name = weight_tensors[i].name
            # Detect problematic layers. Exclude biases of conv layers.
            alert = ""
            if w.min() == w.max() and not (l.__class__.__name__ == "Conv2D" and i == 1):
                alert += "<span style='color:red'>*** dead?</span>"
            if np.abs(w.min()) > 1000 or np.abs(w.max()) > 1000:
                alert += "<span style='color:red'>*** Overflow?</span>"
            # Add row
            table.append([
                weight_name + alert,
                str(w.shape),
                "{:+9.4f}".format(w.min()),
                "{:+10.4f}".format(w.max()),
                "{:+9.4f}".format(w.std()),
            ])
    display_table(table)
