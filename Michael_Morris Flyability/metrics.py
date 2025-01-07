import numpy as np

# Metrics.
def iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) for two bounding boxes.
    Box format: [x, y, width, height] where (x, y) is the center of the box.
    """
    # Convert from center coordinates (x, y, width, height) to corner coordinates (xmin, ymin, xmax, ymax)
    box1_xmin = box1[0] - box1[2] / 2
    box1_ymin = box1[1] - box1[3] / 2
    box1_xmax = box1[0] + box1[2] / 2
    box1_ymax = box1[1] + box1[3] / 2

    box2_xmin = box2[0] - box2[2] / 2
    box2_ymin = box2[1] - box2[3] / 2
    box2_xmax = box2[0] + box2[2] / 2
    box2_ymax = box2[1] + box2[3] / 2

    # Compute the coordinates of the intersection box
    inter_xmin = max(box1_xmin, box2_xmin)
    inter_ymin = max(box1_ymin, box2_ymin)
    inter_xmax = min(box1_xmax, box2_xmax)
    inter_ymax = min(box1_ymax, box2_ymax)

    # Compute area of intersection
    inter_area = max(0, inter_xmax - inter_xmin) * max(0, inter_ymax - inter_ymin)

    # Compute area of both bounding boxes
    box1_area = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)
    box2_area = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)

    # Compute the area of union
    union_area = box1_area + box2_area - inter_area

    # Compute IoU
    iou_value = inter_area / union_area if union_area != 0 else 0
    return iou_value

def mean_average_precision(gt_boxes, pred_boxes, iou_threshold=0.5):
    """
    Calculate Mean Average Precision (mAP) for a list of predictions.
    gt_boxes_list: List of ground truth bounding boxes for each image (N_images, 4)
    pred_boxes_list: List of predicted bounding boxes for each image (N_images, 4)
    """

    ious = [iou(true, pred) for true, pred in zip(gt_boxes, pred_boxes)]
    # True positive if IoU is above the threshold, else false positive
    true_positives = [1 if iou_value >= iou_threshold else 0 for iou_value in ious]
    false_positives = [1 if iou_value < iou_threshold else 0 for iou_value in ious]

    # Calculate precision and recall (For simplicity, we'll calculate Precision at a single threshold)
    precision = np.sum(true_positives) / (np.sum(true_positives) + np.sum(false_positives)) if np.sum(true_positives) > 0 else 0

    return precision