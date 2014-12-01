clear; clc; caffe('set_device',2);
rcnn_model = rcnn_create_model(1,'./model-defs/VGG_ILSVRC_19_layers_batch_1_fc6.prototxt', './data/caffe_nets/VGG_ILSVRC_19_layers.caffemodel');
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'wrap';
rcnn_model.detectors.crop_padding = 16;

for i = 1:3975
  fprintf('Extract Seg Features: #%d\n', i);
  im = imread(['./datasets/MSRC/Seg/5_10/' num2str(i) '.bmp']);
  MSRC_Seg_5_10(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

for i = 1:9185
  fprintf('Extract Seg Features: #%d\n', i);
  im = imread(['./datasets/MSRC/Seg/15_20/' num2str(i) '.bmp']);
  MSRC_Seg_15_20(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

for i = 1:18384
  fprintf('Extract Seg Features: #%d\n', i);
  im = imread(['./datasets/MSRC/Seg/30_40/' num2str(i) '.bmp']);
  MSRC_Seg_30_40(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end
