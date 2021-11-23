_base_ = '../cbnet/faster_rcnn_cbv2d1_r50_fpn_1x_naplab.py'

# Dataset settings
dataset_type = 'CocoDataset'
classes = ('car', 'truck', 'bus', 'motorcycle', 'bicycle', 'scooter', 'person', 'rider')
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='path/to/your/train/annotation_data',
        img_prefix='path/to/your/train/image_data'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='path/to/your/val/annotation_data',
        img_prefix='path/to/your/val/image_data'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='path/to/your/test/annotation_data',
        img_prefix='path/to/your/test/image_data'))

# Model settings
model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=8)))