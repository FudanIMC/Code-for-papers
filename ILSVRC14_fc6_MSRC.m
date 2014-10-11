clear; clc; caffe('set_device',1);
TrainList = importdata('./datasets/MSRC/Train.txt');
TestList = importdata('./datasets/MSRC/Test.txt');
rcnn_model = rcnn_create_model(1,'./model-defs/VGG_ILSVRC_19_layers_batch_1_fc6.prototxt', './data/caffe_nets/VGG_ILSVRC_19_layers.caffemodel');
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'wrap';
rcnn_model.detectors.crop_padding = 16;

for i = 1:276
  fprintf('Extract MSRC Train Features: #%d\n', i);
  im = imread(['./datasets/MSRC/Images/' TrainList{i}]);
  MSRC_TrainX(i,:) = rcnn_features(im, [1,1,size(im,2),size(im,1)], rcnn_model);
end

for i = 1:256
  fprintf('Extract MSRC Test Features: #%d\n', i);
  im = imread(['./datasets/MSRC/Images/' TestList{i}]);
  MSRC_TestX(i,:) = rcnn_features(im, [1,1,size(im,2),size(im,1)], rcnn_model);
end

load('./datasets/MSRC/ImagesDB.mat'); 
MSRC_TrainY = zeros(276,21); MSRC_TestY = zeros(256,21);
for i = 1:276
  Label = ImagesDB{i}.labels;
  for j=2:size(Label,2)
    MSRC_TrainY(i,Label(j)) = 1;
  end
end
for i = 1:256
  Label = ImagesDB{i}.labels;
  for j=2:size(Label,2)
    MSRC_TestY(i,Label(j)) = 1;
  end
end
