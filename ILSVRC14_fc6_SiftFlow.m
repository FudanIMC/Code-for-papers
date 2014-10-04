ImageList = importdata('./datasets/SiftFlow/ImageList.txt');
LabelList = importdata('./datasets/SiftFlow/LabelList.txt');

rcnn_model = rcnn_create_model(1,'./model-defs/VGG_ILSVRC_19_layers_batch_1_fc6.prototxt', './data/caffe_nets/VGG_ILSVRC_19_layers.caffemodel');
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'wrap';
rcnn_model.detectors.crop_padding = 16;

for i = 1:2488
  fprintf('Extract Features: #%d\n', i);
  im = imread(['./datasets/SiftFlow/Images/' ImageList{i}]);
  SiftFlow_TrainX(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

for i = 1:200
  fprintf('Extract Features: #%d\n', i);
  im = imread(['./datasets/SiftFlow/Images/' ImageList{i+2488}]);
  SiftFlow_TestX(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

for k = 1:2488
  load(['./datasets/SiftFlow/SemanticLabels/' LabelList{k}]);
  labels=zeros(1,33);
  for i=1:256
    for j=1:256
      if ((S(i,j)~=0) && (labels(S(i,j))~=1))
        labels(S(i,j))=1;
      end
    end
  end
  SiftFlow_TrainY(k,:)=labels;
end

for k = 1:200
  load(['./datasets/SiftFlow/SemanticLabels/' LabelList{k+2488}]);
  labels=zeros(1,33);
  for i=1:256
    for j=1:256
      if ((S(i,j)~=0) && (labels(S(i,j))~=1))
        labels(S(i,j))=1;
      end
    end
  end
  SiftFlow_TestY(k,:)=labels;
end
