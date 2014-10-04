clear; clc; caffe('set_device',1);
ImageList = importdata('./datasets/VOC2007/Image.txt');
LabelList = importdata('./datasets/VOC2007/Label.txt');
rcnn_model = rcnn_create_model(1,'./model-defs/VGG_ILSVRC_19_layers_batch_1_fc6.prototxt', './data/caffe_nets/VGG_ILSVRC_19_layers.caffemodel');
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'wrap';
rcnn_model.detectors.crop_padding = 16;

for i = 1:422
  fprintf('Extract Features: #%d\n', i);
  im = imread(['./datasets/VOC2007/Images/' ImageList{i}]);
  VOC2007_TrainX(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

for i = 1:210
  fprintf('Extract Features: #%d\n', i);
  im = imread(['./datasets/VOC2007/Images/' ImageList{i+422}]);
  VOC2007_TestX(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

for k = 1:422
  im=imread(['./datasets/VOC2007/SemanticLabels/' LabelList{k}]);
  labels=zeros(1,21); imsize=size(im);
  for i=1:imsize(1)
    for j=1:imsize(2)
      if (labels(im(i,j)+1)~=1)
        labels(im(i,j)+1)=1;
      end
    end
  end
  VOC2007_TrainY(k,:)=labels;
end

for k = 1:210
  im=imread(['./datasets/VOC2007/SemanticLabels/' LabelList{k+422}]);
  labels=zeros(1,21); imsize=size(im);
  for i=1:imsize(1)
    for j=1:imsize(2)
      if (labels(im(i,j)+1)~=1)
        labels(im(i,j)+1)=1;
      end
    end
  end
  VOC2007_TestY(k,:)=labels;
end
