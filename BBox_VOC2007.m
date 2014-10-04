clear; close all; clc; load('VOC2007_Regions.mat');
List = importdata('./datasets/VOC2007/List.txt');

caffe('set_device',2); BBox=[];
rcnn_model = rcnn_create_model(55,'./model-defs/VGG_ILSVRC_19_layers_batch_55_fc6.prototxt', './data/caffe_nets/VGG_ILSVRC_19_layers.caffemodel');
rcnn_model = rcnn_load_model(rcnn_model);
rcnn_model.detectors.crop_mode = 'wrap';
rcnn_model.detectors.crop_padding = 16;

% Test the 'fast' version, which takes around 5 seconds in mean
% [candidates_mcg, ucm2_mcg] = im2mcg(im,'fast');

% Test the 'accurate' version, which tackes around 30 seconds in mean
% [candidates_mcg, ucm2_mcg] = im2mcg(im,'accurate');

for i=1:632
  im = imread(['./datasets/VOC2007/Images/' List{i} '.png']); 
  fprintf('Extract %d Train Feature\n',i); 
  load(['./datasets/VOC2007/MCG/' List{i} '.mat']);
  X_tst = rcnn_features(im, boxes, rcnn_model); 
  Y_tst = ones(1,size(X_tst,1))'; bboxes = []; 
  feat = rcnn_features(im, boxes, rcnn_model);
  for k=2:21
    if (VOC2007_TrainY(i,k))
      [X,Y,Z] = predict(Y_tst,sparse(double(X_tst)),SVM_model(k));
      Regions = [boxes,X]; Regions = sortrows(Regions,-5); 
      N_Regions = size(find(Regions(:,5)==1),1);
      bbox = vl_gmm(Regions(1:N_Regions,1:4)',1)';
      bboxes = [bboxes;[i,bbox,k]];
      %showboxes(im,bbox(1,:),'g'); 
    end
  end
  BBox = [BBox; bboxes];
end
save(['./datasets/VOC2007/BBox.mat'],'','BBox','-v7.3');
