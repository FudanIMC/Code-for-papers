ImageList = importdata('./datasets/SiftFlow/list.txt');

rcnn_model = rcnn_create_model('./model-defs/rcnn_batch_256_output_pool5.prototxt', './data/caffe_nets/finetune_ilsvrc13_val1+train1k_iter_50000');
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
  im = imread(['./datasets/SiftFlow/Images/' ImageList{i}]);
  SiftFlow_TestX(i,:) = rcnn_features(im, [1,1,size(im)], rcnn_model);
end

LabelList = importdata('./datasets/SiftFlow/LabelList.txt');
SiftFlow_TrainY=zeros(2488,33);

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
